def compute_recall(results_file_path):
    recall_at_1 = 0
    recall_at_3 = 0
    total_queries = 0

    with open(results_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue  # skip malformed lines

            query = parts[0]
            top3_matches = parts[1:4]

            # Extract expected match from query filename
            expected_match = query.split('-snippet')[0].replace('.wav', '') + '.wav'

            if expected_match == top3_matches[0]:
                recall_at_1 += 1
            if expected_match in top3_matches:
                recall_at_3 += 1

            total_queries += 1

    r1 = recall_at_1 / total_queries if total_queries > 0 else 0
    r3 = recall_at_3 / total_queries if total_queries > 0 else 0

    return r1, r3, recall_at_1, recall_at_3, total_queries


if __name__ == "__main__":
    r1, r3, r1_count, r3_count, total = compute_recall('results.txt')
    print(f"Recall@1: {r1:.2%} ({r1_count}/{total})")
    print(f"Recall@3: {r3:.2%} ({r3_count}/{total})")