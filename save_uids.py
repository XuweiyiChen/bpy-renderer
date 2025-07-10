#!/usr/bin/env python3
"""
Simple script to save the first 1000 Objaverse UIDs to a text file.
Each UID will be on a separate line.
"""

import objaverse

def save_uids_to_file(filename="uids_1000.txt", num_uids=1000):
    """
    Save the first num_uids UIDs to a text file.
    
    Args:
        filename (str): Output filename
        num_uids (int): Number of UIDs to save
    """
    
    print("Loading Objaverse UIDs...")
    uids = objaverse.load_uids()
    
    print(f"Total UIDs available: {len(uids):,}")
    
    # Get the first num_uids
    uids_subset = uids[:num_uids]
    
    # Save to file
    with open(filename, 'w') as f:
        for uid in uids_subset:
            f.write(uid + '\n')
    
    print(f"Saved {len(uids_subset)} UIDs to '{filename}'")
    print(f"First 5 UIDs:")
    for i, uid in enumerate(uids_subset[:5]):
        print(f"  {i+1}. {uid}")
    print("...")
    print(f"Last 5 UIDs:")
    for i, uid in enumerate(uids_subset[-5:], len(uids_subset)-4):
        print(f"  {i}. {uid}")

if __name__ == "__main__":
    save_uids_to_file() 