import pickle
import os

def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def save_pickle(data, filename):
    #os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    print("Loading data files...")
    
    # Load v36 data
    matrices_v36 = load_pickle('genetic-gm')
    heights_v36 = load_pickle('genetic-mh')
    
    # Load memetic-2 data
    matrices_mem2 = load_pickle('generatorMatrix-combined')
    heights_mem2 = load_pickle('mHeight-combined')
    
    combined_matrices = {}
    combined_heights = {}
    
    # Get all unique (n, k, m) targets from both sets
    all_keys = set(heights_v36.keys()).union(set(heights_mem2.keys()))
    
    print("\n--- Merging Results ---")
    for key in sorted(all_keys):
        h_v36 = heights_v36.get(key, float('inf'))
        h_mem2 = heights_mem2.get(key, float('inf'))
        print(f"Comparing target {key}: v36 m-height = {h_v36:.4f}, memetic-2 m-height = {h_mem2:.4f}")

        # Compare and take the best (lowest) score
        if h_v36 < h_mem2:
            combined_heights[key] = h_v36
            combined_matrices[key] = matrices_v36[key]
            winner = "v36"
        else:
            combined_heights[key] = h_mem2
            combined_matrices[key] = matrices_mem2[key]
            winner = "memetic-2"
            
        print(f"Target {key}: Kept {combined_heights[key]:.4f} (from {winner})")

    # Save the combined dictionaries
    print("\nSaving combined dictionaries...")
    save_pickle(combined_matrices, 'generatorMatrix-BEST')
    save_pickle(combined_heights, 'mHeight-BEST')
    
    print("Done! Data safely merged into 'generatorMatrix-BEST' and 'mHeight-BEST'.")