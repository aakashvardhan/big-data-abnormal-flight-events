"""
Sequence feature engineering for flight anomaly detection.
Extracts event sequence patterns, n-grams, and state transitions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict


def extract_sequence_features(flight_df: pd.DataFrame,
                               all_flights_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Extract sequence-based features from a single flight's event sequence.
    
    Args:
        flight_df: DataFrame for a single flight, sorted by timestamp
        all_flights_df: Optional DataFrame of all flights for frequency calculations
        
    Returns:
        Dictionary of sequence features
    """
    if len(flight_df) == 0:
        return {}
    
    features = {}
    
    # Ensure events are sorted by timestamp
    flight_df = flight_df.sort_values('timestamp').reset_index(drop=True)
    event_sequence = flight_df['event_type'].tolist()
    
    # Basic sequence features
    features.update(_extract_basic_sequence_features(event_sequence))
    
    # N-gram features
    features.update(_extract_ngram_features(event_sequence, n=2))  # Bigrams
    features.update(_extract_ngram_features(event_sequence, n=3))  # Trigrams
    
    # State transition features
    features.update(_extract_transition_features(event_sequence))
    
    # Rare sequence patterns
    if all_flights_df is not None:
        features.update(_extract_rare_pattern_features(event_sequence, all_flights_df))
    
    # Sequence complexity
    features.update(_extract_sequence_complexity(event_sequence))
    
    return features


def _extract_basic_sequence_features(event_sequence: List[str]) -> Dict:
    """Extract basic sequence statistics."""
    features = {}
    
    features['sequence_length'] = len(event_sequence)
    features['unique_events_in_sequence'] = len(set(event_sequence))
    features['sequence_diversity'] = features['unique_events_in_sequence'] / max(features['sequence_length'], 1)
    
    # Count consecutive identical events
    consecutive_same = 0
    max_consecutive_same = 0
    current_consecutive = 1
    
    for i in range(1, len(event_sequence)):
        if event_sequence[i] == event_sequence[i-1]:
            current_consecutive += 1
            consecutive_same += 1
            max_consecutive_same = max(max_consecutive_same, current_consecutive)
        else:
            current_consecutive = 1
    
    features['num_consecutive_identical_events'] = consecutive_same
    features['max_consecutive_identical'] = max_consecutive_same
    
    # Event repetition rate
    event_counts = Counter(event_sequence)
    features['most_frequent_event_count'] = max(event_counts.values()) if event_counts else 0
    features['event_repetition_rate'] = features['most_frequent_event_count'] / max(len(event_sequence), 1)
    
    return features


def _extract_ngram_features(event_sequence: List[str], n: int = 2) -> Dict:
    """Extract n-gram features."""
    features = {}
    
    if len(event_sequence) < n:
        features[f'num_unique_{n}grams'] = 0
        features[f'{n}gram_diversity'] = 0
        features[f'most_common_{n}gram_count'] = 0
        return features
    
    # Generate n-grams
    ngrams = []
    for i in range(len(event_sequence) - n + 1):
        ngram = tuple(event_sequence[i:i+n])
        ngrams.append(ngram)
    
    ngram_counts = Counter(ngrams)
    
    features[f'num_unique_{n}grams'] = len(ngram_counts)
    features[f'{n}gram_diversity'] = len(ngram_counts) / max(len(ngrams), 1)
    features[f'most_common_{n}gram_count'] = max(ngram_counts.values()) if ngram_counts else 0
    
    # Store most common n-gram (as string representation)
    if ngram_counts:
        most_common = ngram_counts.most_common(1)[0]
        features[f'most_common_{n}gram'] = ' -> '.join(most_common[0])
    else:
        features[f'most_common_{n}gram'] = None
    
    return features


def _extract_transition_features(event_sequence: List[str]) -> Dict:
    """Extract state transition features."""
    features = {}
    
    if len(event_sequence) < 2:
        features['num_unique_transitions'] = 0
        features['transition_diversity'] = 0
        features['num_self_transitions'] = 0
        features['self_transition_rate'] = 0
        return features
    
    # Calculate transitions
    transitions = []
    self_transitions = 0
    
    for i in range(1, len(event_sequence)):
        transition = (event_sequence[i-1], event_sequence[i])
        transitions.append(transition)
        if transition[0] == transition[1]:
            self_transitions += 1
    
    transition_counts = Counter(transitions)
    
    features['num_unique_transitions'] = len(transition_counts)
    features['transition_diversity'] = len(transition_counts) / max(len(transitions), 1)
    features['num_self_transitions'] = self_transitions
    features['self_transition_rate'] = self_transitions / max(len(transitions), 1)
    
    # Most common transition
    if transition_counts:
        most_common = transition_counts.most_common(1)[0]
        features['most_common_transition'] = f"{most_common[0][0]} -> {most_common[0][1]}"
        features['most_common_transition_count'] = most_common[1]
    else:
        features['most_common_transition'] = None
        features['most_common_transition_count'] = 0
    
    # Calculate transition entropy (measure of randomness)
    if len(transition_counts) > 0:
        total_transitions = sum(transition_counts.values())
        probabilities = [count / total_transitions for count in transition_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        features['transition_entropy'] = entropy
        features['max_possible_entropy'] = np.log2(len(transition_counts)) if len(transition_counts) > 0 else 0
        features['normalized_entropy'] = entropy / max(features['max_possible_entropy'], 1e-10)
    else:
        features['transition_entropy'] = 0
        features['max_possible_entropy'] = 0
        features['normalized_entropy'] = 0
    
    return features


def _extract_rare_pattern_features(event_sequence: List[str],
                                   all_flights_df: pd.DataFrame) -> Dict:
    """Extract features related to rare event patterns."""
    features = {}
    
    # Calculate n-gram frequencies across all flights
    all_sequences = []
    for flight_id in all_flights_df['flight_id'].unique():
        flight_events = all_flights_df[all_flights_df['flight_id'] == flight_id]
        flight_events = flight_events.sort_values('timestamp')
        all_sequences.append(flight_events['event_type'].tolist())
    
    # Calculate bigram frequencies
    all_bigrams = []
    for seq in all_sequences:
        for i in range(len(seq) - 1):
            all_bigrams.append((seq[i], seq[i+1]))
    
    bigram_freq = Counter(all_bigrams)
    total_bigrams = sum(bigram_freq.values())
    
    # Count rare bigrams in this flight
    flight_bigrams = []
    for i in range(len(event_sequence) - 1):
        flight_bigrams.append((event_sequence[i], event_sequence[i+1]))
    
    rare_threshold = 0.01  # Bigrams that occur in < 1% of all transitions
    rare_bigrams = 0
    
    for bigram in set(flight_bigrams):
        freq = bigram_freq.get(bigram, 0) / max(total_bigrams, 1)
        if freq < rare_threshold:
            rare_bigrams += 1
    
    features['num_rare_bigrams'] = rare_bigrams
    features['rare_bigram_rate'] = rare_bigrams / max(len(set(flight_bigrams)), 1)
    
    # Check for completely unique bigrams (never seen before)
    unique_bigrams = sum(1 for bg in set(flight_bigrams) if bigram_freq.get(bg, 0) == 0)
    features['num_unique_bigrams'] = unique_bigrams
    
    return features


def _extract_sequence_complexity(event_sequence: List[str]) -> Dict:
    """Calculate sequence complexity metrics."""
    features = {}
    
    # Sequence complexity score
    complexity = 0
    
    # Factor 1: Number of unique events
    complexity += len(set(event_sequence))
    
    # Factor 2: Sequence length (longer sequences are more complex)
    complexity += len(event_sequence) / 10  # Normalize
    
    # Factor 3: Number of unique transitions
    if len(event_sequence) > 1:
        transitions = set((event_sequence[i-1], event_sequence[i]) 
                          for i in range(1, len(event_sequence)))
        complexity += len(transitions)
    
    features['sequence_complexity_score'] = complexity
    
    # Pattern regularity (lower = more regular/repetitive)
    if len(event_sequence) > 1:
        # Calculate how often the sequence repeats itself
        event_counts = Counter(event_sequence)
        max_count = max(event_counts.values())
        regularity = max_count / len(event_sequence)
        features['pattern_regularity'] = regularity
        features['pattern_irregularity'] = 1 - regularity
    else:
        features['pattern_regularity'] = 1.0
        features['pattern_irregularity'] = 0.0
    
    return features


def extract_sequence_features_batch(flight_summary_df: pd.DataFrame,
                                     events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sequence features for all flights in batch.
    
    Args:
        flight_summary_df: Flight-level summary DataFrame
        events_df: Event-level DataFrame (sorted by flight and timestamp)
        
    Returns:
        DataFrame with sequence features added
    """
    print("Extracting sequence features for all flights...")
    
    sequence_features_list = []
    
    # Extract features for each flight
    for flight_id in flight_summary_df['flight_id'].unique():
        flight_events = events_df[events_df['flight_id'] == flight_id].copy()
        
        if len(flight_events) > 0:
            features = extract_sequence_features(flight_events, events_df)
            features['flight_id'] = flight_id
            sequence_features_list.append(features)
    
    sequence_df = pd.DataFrame(sequence_features_list)
    
    # Merge with flight summary
    result_df = flight_summary_df.merge(sequence_df, on='flight_id', how='left')
    
    print(f"✓ Extracted sequence features for {len(result_df)} flights")
    
    return result_df



