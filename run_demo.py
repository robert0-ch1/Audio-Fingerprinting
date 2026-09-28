import os
from audio_identification import fingerprintBuilder, audioIdentification

# Ensure directory paths exist
os.makedirs("fingerprints", exist_ok=True)

# Define paths
database_path = "./database"
fingerprints_path = "./fingerprints"
queryset_path = "./queryset"
output_path = "./results.txt"

# Run the fingerprint builder
print("\n=== BUILDING FINGERPRINTS ===")
print(f"Reading audio files from: {database_path}")
print(f"Saving fingerprints to: {fingerprints_path}\n")
fingerprintBuilder(database_path, fingerprints_path)

# Run the audio identification
print("\n=== IDENTIFYING AUDIO ===")
print(f"Processing queries from: {queryset_path}")
print(f"Using fingerprints from: {fingerprints_path}")
print(f"Saving results to: {output_path}\n")
audioIdentification(queryset_path, fingerprints_path, output_path)

print("\n=== PROCESS COMPLETE ===")
print(f"Results saved to: {output_path}")
print("You can view the matches for each query in this file.")
