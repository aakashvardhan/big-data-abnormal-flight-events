# Alpine-based JDK8 that runs on arm64 cleanly
FROM alpine:3.19

ARG KAFKA_VERSION=2.3.0
ARG SCALA_VERSION=2.12
ENV KAFKAHOME=/home
# PATH for bash sessions
ENV PATH=$KAFKAHOME/kafka/bin:$PATH

# Basic tools + bash
RUN apk add --no-cache bash curl tar coreutils openjdk8-jdk


# Download Kafka 2.3.0 (Scala 2.12) from Apache archives
RUN mkdir -p ${KAFKAHOME} \
    && curl -fSL https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz -o /tmp/kafka.tgz \
    && tar -xzf /tmp/kafka.tgz -C ${KAFKAHOME} \
    && mv ${KAFKAHOME}/kafka_${SCALA_VERSION}-${KAFKA_VERSION} ${KAFKAHOME}/kafka \
    && chmod +x ${KAFKAHOME}/kafka/bin/*.sh \
    && rm -f /tmp/kafka.tgz
# Required data dirs
RUN mkdir -p ${KAFKAHOME}/kafka/data/zookeeper ${KAFKAHOME}/kafka/data/server


# Configure ZK dataDir and broker log.dirs; wire ZK hostname for compose
RUN sed -i 's|^dataDir=.*|dataDir=/home/kafka/data/zookeeper|' ${KAFKAHOME}/kafka/config/zookeeper.properties \
    && sed -i 's|^zookeeper.connect=.*|zookeeper.connect=zookeeper:2181|' ${KAFKAHOME}/kafka/config/server.properties \
    && sed -i 's|^#\?log.dirs=.*|log.dirs=/home/kafka/data/server|' ${KAFKAHOME}/kafka/config/server.properties

# define two listeners and map protocols
# you need two listeners: one so container inside Docker talk to 'broker:9092'
# and another so my local machine connects through 'localhost:29092'
# allows flexibility in running python scripts
RUN sed -i 's|^#\?listeners=.*||' ${KAFKAHOME}/kafka/config/server.properties \
    && sed -i 's|^#\?advertised.listeners=.*||' ${KAFKAHOME}/kafka/config/server.properties \
    && sed -i 's|^#\?inter.broker.listener.name=.*||' ${KAFKAHOME}/kafka/config/server.properties \
    && sed -i 's|^#\?listener.security.protocol.map=.*||' ${KAFKAHOME}/kafka/config/server.properties \
    && printf '\n%s\n' \
    'listeners=PLAINTEXT://0.0.0.0:9092,PLAINTEXT_HOST://0.0.0.0:29092' \
    'advertised.listeners=PLAINTEXT://broker:9092,PLAINTEXT_HOST://localhost:29092' \
    'listener.security.protocol.map=PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT' \
    'inter.broker.listener.name=PLAINTEXT' \
    >> ${KAFKAHOME}/kafka/config/server.properties


# Add a non-root user with a home so .bashrc exists
RUN adduser -D -h /home/kafka kafka \
    && mkdir -p /home/kafka \
    && echo 'export KAFKAHOME=/home' >> /home/kafka/.bashrc \
    && echo 'export PATH=$KAFKAHOME/kafka/bin:$PATH' >> /home/kafka/.bashrc \
    && chown -R kafka:kafka /home/kafka


# create a startup shim to clear stale lock and start Kafka cleanly
RUN printf '%s\n' '#!/bin/sh' \
    'rm -f /home/kafka/data/server/.lock' \
    'exec /home/kafka/bin/kafka-server-start.sh /home/kafka/config/server.properties' \
    > /usr/local/bin/kafka-start \
    && chmod +x /usr/local/bin/kafka-start


USER kafka
WORKDIR /home/kafka
