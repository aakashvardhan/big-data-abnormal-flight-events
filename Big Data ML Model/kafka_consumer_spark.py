# kafka_consumer_spark.py
"""
Kafka Consumer with Spark Structured Streaming: Real-time flight delay predictions
Usage: spark-submit kafka_consumer_spark.py --topic flight-data --model-path flight_delay_rf_model
"""

import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf


# Define UDF for extracting delay probability
@udf(returnType=DoubleType())
def get_delay_prob(probability):
    return float(probability[1]) if probability else 0.0


def create_spark_session(app_name="FlightDelayPrediction"):
    """Create and configure Spark session"""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.maxResultSize", "1g") \
        .config("spark.sql.streaming.checkpointLocation", "file:///tmp/kafka_checkpoint") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0") \
        .getOrCreate()


def load_model(spark, model_path):
    """Load trained Random Forest model"""
    abs_path = f"file://{os.path.abspath(model_path)}"
    model = RandomForestClassificationModel.load(abs_path)
    print(f"✓ Model loaded from {abs_path}")
    return model


def preprocess_streaming_data(df):
    """
    Preprocess streaming flight data - simplified for streaming
    
    Args:
        df: Spark streaming DataFrame with raw flight data
    
    Returns:
        Preprocessed DataFrame with features
    """
    # Define feature columns
    numerical_features = [
        'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
        'CRS_DEP_TIME', 'CRS_ARR_TIME',
        'DISTANCE', 'CRS_ELAPSED_TIME'
    ]
    
    categorical_features = [
        'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST'
    ]
    
    # Type conversion
    int_cols = ["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "CRS_DEP_TIME", "CRS_ARR_TIME"]
    float_cols = ["DISTANCE", "CRS_ELAPSED_TIME"]
    
    for col in int_cols:
        if col in df.columns:
            df = df.withColumn(col, F.expr(f"try_cast({col} as int)"))
    
    for col in float_cols:
        if col in df.columns:
            df = df.withColumn(col, F.expr(f"try_cast({col} as double)"))
    
    # Select features and drop nulls
    df = df.select(numerical_features + categorical_features).dropna()
    
    # Simple encoding: use hash for categorical features (no fitting needed)
    for col in categorical_features:
        df = df.withColumn(f"{col}_indexed", F.abs(F.hash(F.col(col))) % 10000)
    
    # Assemble features
    indexed_categorical = [f"{col}_indexed" for col in categorical_features]
    all_features = numerical_features + indexed_categorical
    
    assembler = VectorAssembler(
        inputCols=all_features,
        outputCol="features",
        handleInvalid="skip"
    )
    
    df = assembler.transform(df)
    
    return df


def predict_and_output(batch_df, batch_id, model, threshold=0.5):
    """
    Process each micro-batch: make predictions and output high-risk flights
    
    Args:
        batch_df: Micro-batch DataFrame
        batch_id: Batch identifier
        model: Trained ML model
        threshold: Probability threshold for delay prediction (default: 0.5)
    """
    if batch_df.isEmpty():
        return
    
    print(f"\n📥 Processing batch {batch_id}...")
    
    try:
        # Preprocess
        processed_df = preprocess_streaming_data(batch_df)
        
        if processed_df.count() == 0:
            print(f"⚠️  No valid records in batch {batch_id}")
            return
        
        # Make predictions
        predictions = model.transform(processed_df)
        
        # Extract delay probability
        predictions = predictions.withColumn(
            "delay_probability",
            get_delay_prob(F.col("probability"))
        )
        
        # Diagnostic: Show prediction distribution
        #pred_counts = predictions.groupBy("prediction").count().collect()
        #print(f"  Prediction distribution: {dict((row['prediction'], row['count']) for row in pred_counts)}")
        
        # Count records and show probability statistics
        record_count = predictions.count()
        prob_stats = predictions.agg(
            F.min("delay_probability").alias("min_prob"),
            F.max("delay_probability").alias("max_prob"),
            F.avg("delay_probability").alias("avg_prob")
        ).collect()[0]
        print(f"  Records: {record_count}, Delay probability - min: {prob_stats['min_prob']:.3f}, max: {prob_stats['max_prob']:.3f}, avg: {prob_stats['avg_prob']:.3f}")
        
        # Get highest risk flight
        highest_risk = predictions.orderBy(F.desc("delay_probability")).limit(1)
        
        # Select output columns with custom threshold
        output = highest_risk.select(
            F.col("OP_UNIQUE_CARRIER").alias("carrier"),
            F.col("ORIGIN").alias("origin"),
            F.col("DEST").alias("destination"),
            F.col("CRS_DEP_TIME").alias("scheduled_departure"),
            F.when(F.col("delay_probability") >= threshold, 1).otherwise(0).alias("prediction"),
            F.col("delay_probability"))
        
        # Convert to JSON and print
        result = output.toJSON().collect()
        for json_str in result:
            print(json_str)
        
    except Exception as e:
        print(f"❌ Error processing batch {batch_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Real-time flight delay prediction with Kafka')
    parser.add_argument(
        '--bootstrap-servers',
        default='localhost:9092',
        help='Kafka bootstrap servers (default: localhost:9092)'
    )
    parser.add_argument(
        '--topic',
        default='flight-data',
        help='Kafka topic to consume from (default: flight-data)'
    )
    parser.add_argument(
        '--model-path',
        default='flight_delay_rf_model',
        help='Path to trained model (default: flight_delay_rf_model)'
    )
    parser.add_argument(
        '--output-mode',
        default='append',
        choices=['append', 'update', 'complete'],
        help='Streaming output mode (default: append)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.45,
        help='Probability threshold for delay prediction (default: 0.45)'
    )
    parser.add_argument(
        '--starting-offset',
        default='earliest',
        choices=['earliest', 'latest'],
        help='Where to start reading messages (default: earliest)'
    )
    
    args = parser.parse_args()
    
    # Create Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Load model
    model = load_model(spark, args.model_path)
    
    # Read from Kafka with consumer group for offset tracking
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", args.bootstrap_servers) \
        .option("subscribe", args.topic) \
        .option("kafka.group.id", "flight-delay-consumer-group") \
        .option("startingOffsets", args.starting_offset) \
        .option("failOnDataLoss", "false") \
        .load()
    
    # Parse JSON from Kafka
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
    
    # Define schema for flight data JSON
    flight_schema = StructType([
        StructField("OP_UNIQUE_CARRIER", StringType(), True),
        StructField("ORIGIN", StringType(), True),
        StructField("DEST", StringType(), True),
        StructField("CRS_DEP_TIME", StringType(), True),
        StructField("CRS_ARR_TIME", StringType(), True),
        StructField("CRS_ELAPSED_TIME", StringType(), True),
        StructField("DISTANCE", StringType(), True),
        StructField("MONTH", StringType(), True),
        StructField("DAY_OF_MONTH", StringType(), True),
        StructField("DAY_OF_WEEK", StringType(), True)
    ])
    
    parsed_df = df.selectExpr("CAST(value AS STRING) as json") \
        .select(F.from_json(F.col("json"), flight_schema).alias("data")) \
        .select("data.*")
    
    print(f"✓ Kafka consumer started. Listening for flight data...")
    print(f"  Using threshold: {args.threshold}")
    print(f"  Starting offset: {args.starting_offset}")
    
    # Process streaming data
    query = parsed_df \
        .writeStream \
        .foreachBatch(lambda batch_df, batch_id: predict_and_output(batch_df, batch_id, model, args.threshold)) \
        .outputMode(args.output_mode) \
        .start()
    
    query.awaitTermination()


if __name__ == "__main__":
    main()
