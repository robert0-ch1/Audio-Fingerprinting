# === AUDIO IDENTIFICATION SYSTEM ===
# Shazam-style audio fingerprinting and identification

import os
import pickle
import numpy as np
import librosa
from scipy import ndimage
from collections import defaultdict, Counter

# Global parameters for the audio fingerprinting system
SAMPLE_RATE = 22050  # Base sample rate for processing
N_FFT = 1024         # FFT window size
HOP_LENGTH = 256     # Hop length for STFT
PEAK_NEIGHBORHOOD_SIZE = 15  # Size of neighborhood for peak finding
MIN_PEAK_AMPLITUDE = 0.08    # Minimum amplitude for peak detection
MAX_PEAKS_PER_FRAME = 6      # Maximum number of peaks per time frame
TARGET_DENSITY = 35          # Target density for peak constellation
FAN_OUT = 20                 # Number of pairs to generate per anchor point
MIN_TIME_DELTA = 1           # Minimum time delta between paired peaks
MAX_TIME_DELTA = 150         # Maximum time delta between paired peaks


def extract_constellation(y, sr):
    """
    Extract spectrogram peaks using STFT transform.
    """
    # Generate spectrogram
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    
    # Convert to dB scale
    D_log = librosa.amplitude_to_db(D, ref=np.max)
    
    # Normalize
    D_norm = (D_log - np.min(D_log)) / (np.max(D_log) - np.min(D_log))
    
    # Apply local maximum filter
    local_max = ndimage.maximum_filter(D_norm, size=PEAK_NEIGHBORHOOD_SIZE) == D_norm
    
    # Apply amplitude threshold
    amplitude_thresh = D_norm > MIN_PEAK_AMPLITUDE
    detected_peaks = local_max & amplitude_thresh
    
    # Get coordinates of peaks
    peak_coords = np.argwhere(detected_peaks)
    
    # Sort by amplitude and select top peaks per frame
    freqs, times = peak_coords[:, 0], peak_coords[:, 1]
    amplitudes = D_norm[freqs, times]
    
    # Group by time frame
    frame_groups = defaultdict(list)
    for i, (f, t, a) in enumerate(zip(freqs, times, amplitudes)):
        frame_groups[t].append((f, a, i))
    
    # Select top peaks per frame based on amplitude
    selected_peaks = []
    for t, peaks in frame_groups.items():
        # Sort peaks in this frame by amplitude (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)
        # Select the top N peaks
        for f, a, i in peaks[:MAX_PEAKS_PER_FRAME]:
            selected_peaks.append((f, t))
    
    # Control overall constellation density
    if len(selected_peaks) > TARGET_DENSITY * (len(frame_groups) / 4):
        # Take peaks with highest amplitude
        sorted_peaks = sorted([(f, t, D_norm[f, t]) for f, t in selected_peaks], 
                             key=lambda x: x[2], reverse=True)
        num_to_keep = int(TARGET_DENSITY * (len(frame_groups) / 4))
        selected_peaks = [(f, t) for f, t, _ in sorted_peaks[:num_to_keep]]
    
    return selected_peaks


def generate_hashes(constellation):
    """
    Generate combinatorial hashes using a target zone approach.
    Converts the hashes to 32-bit integers for storage efficiency.
    """
    hashes = []
    for i, (f1, t1) in enumerate(constellation):
        # Define the target zone for this anchor point
        # Look at points ahead in time, limited by max_time_delta
        j = i + 1
        target_points = 0
        
        while j < len(constellation) and target_points < FAN_OUT:
            f2, t2 = constellation[j]
            # Check if point is within the target zone (time constraint)
            time_delta = t2 - t1
            if time_delta < MIN_TIME_DELTA:
                j += 1
                continue
            if time_delta > MAX_TIME_DELTA:
                break
                
            # Create 32-bit hash
            # Format: combines frequency components and time delta
            # f1: 10 bits, f2: 10 bits, delta_t: 12 bits
            f1_bits = min(f1, 1023)  # Cap at 10 bits
            f2_bits = min(f2, 1023)  # Cap at 10 bits
            dt_bits = min(time_delta, 4095)  # Cap at 12 bits
            
            # Pack into a 32-bit hash
            hash_value = (f1_bits << 22) | (f2_bits << 12) | dt_bits
            
            hashes.append((hash_value, t1))
            target_points += 1
            j += 1
    
    return hashes


def extract_hashes_from_file(audio_path):
    """Extract fingerprint hashes from a single audio file."""
    try:
        # For query files, they might be at 44100Hz
        y, sr = librosa.load(audio_path, sr=None)
        
        # Resample to target sample rate if needed
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Extract constellation and generate hashes
        constellation = extract_constellation(y, SAMPLE_RATE)
        hashes = generate_hashes(constellation)
        
        return hashes
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return []


def build_hash_index(file_hashes):
    """Build a searchable index of hashes from the database files."""
    index = defaultdict(list)
    for file, hashes in file_hashes.items():
        for h, t in hashes:
            index[h].append((file, t))
    return index


def match_query(query_hashes, db_index, k=3, min_cluster_size=3, bin_width=2):
    """Return top-k matching database files."""
    # Track offset histograms for each file
    offset_counter = defaultdict(Counter)
    
    # Count matches at different time offsets
    for query_hash, t_query in query_hashes:
        for db_file, t_db in db_index.get(query_hash, []):
            offset = t_db - t_query
            binned_offset = offset // bin_width
            offset_counter[db_file][binned_offset] += 1
    
    # Score each file based on strongest alignment cluster
    scored_matches = []
    for file, offsets in offset_counter.items():
        if not offsets:
            continue
        # Get the strongest alignment
        top_offset, count = offsets.most_common(1)[0]
        
        # Calculate confidence metrics
        match_ratio = count / max(1, len(query_hashes))
        confidence = count * np.log10(1 + count)
        
        scored_matches.append({
            'file': file,
            'score': count,
            'confidence': confidence,
            'match_ratio': match_ratio,
            'offset': top_offset * bin_width
        })
    
    # Sort by score (descendingly)
    scored_matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top k matches or fewer if not enough matches
    return [match['file'] for match in scored_matches[:k]]


def fingerprintBuilder(database_path, fingerprints_path):
    """
    Build fingerprints from audio files in the database and save them.
    
    Args:
        database_path: Path to the folder containing database audio files
        fingerprints_path: Path where fingerprints will be saved
    """
    print(f"Building fingerprints from {database_path}...")
    
    # Create fingerprints directory
    os.makedirs(fingerprints_path, exist_ok=True)
    
    # Extract fingerprints for all database files
    database_hashes = {}
    file_count = 0
    
    # Process all audio files in the database folder recursively
    for root, _, files in os.walk(database_path):
        for filename in files:
            if filename.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, filename)
                # Use relative path as the key 
                rel_path = os.path.relpath(file_path, database_path)
                
                # Extract hashes
                hashes = extract_hashes_from_file(file_path)
                if hashes:
                    database_hashes[rel_path] = hashes
                    file_count += 1
                    if file_count % 10 == 0:
                        print(f"Processed {file_count} files...")
    
    print(f"Extracted fingerprints from {file_count} audio files")
    
    # Build and save the hash index
    hash_index = build_hash_index(database_hashes)
    
    # Save the fingerprints and index
    with open(os.path.join(fingerprints_path, 'db_hashes.pkl'), 'wb') as f:
        pickle.dump(database_hashes, f)
    
    with open(os.path.join(fingerprints_path, 'hash_index.pkl'), 'wb') as f:
        pickle.dump(hash_index, f)
    
    print(f"Fingerprints saved to {fingerprints_path}")


def audioIdentification(queryset_path, fingerprints_path, output_path):
    """
    Identify audio files from the query set using the pre-built fingerprints.
    
    Args:
        queryset_path: Path to the folder containing query audio files
        fingerprints_path: Path where fingerprints are stored
        output_path: Path where results will be saved
    """
    print(f"Identifying audio from {queryset_path}...")
    
    # Load fingerprints and index
    try:
        with open(os.path.join(fingerprints_path, 'hash_index.pkl'), 'rb') as f:
            hash_index = pickle.load(f)
        
        print("Loaded fingerprint database")
    except FileNotFoundError:
        print(f"Error: Fingerprint database not found in {fingerprints_path}")
        return
    
    # Process all query files
    results = []
    file_count = 0
    
    # Process all audio files in the query folder recursively
    for root, _, files in os.walk(queryset_path):
        for filename in sorted(files):
            if filename.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, queryset_path)
                
                # Extract hashes from query
                query_hashes = extract_hashes_from_file(file_path)
                
                # Match against database
                matches = match_query(query_hashes, hash_index, k=3)
                
                # Ensure we have exactly 3 matches
                while len(matches) < 3:
                    matches.append("")
                
                # Store result
                results.append((rel_path, matches))
                file_count += 1
                if file_count % 10 == 0:
                    print(f"Processed {file_count} queries...")
    
    # Write results to output file
    with open(output_path, 'w') as f:
        for query, matches in results:
            line = query + '\t' + '\t'.join(matches[:3]) + '\n'
            f.write(line)
    
    print(f"Identification complete. Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    # fingerprintBuilder("./database", "./fingerprints")
    # audioIdentification("./queryset", "./fingerprints", "./results.txt")
    
    # Please uncomment and modify the paths above to further test the system
    print("Run this module with the following functions:")
    print("fingerprintBuilder(database_path, fingerprints_path)")
    print("audioIdentification(queryset_path, fingerprints_path, output_path)")
