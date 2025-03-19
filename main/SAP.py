"""
Separate implementations of SAP (Sweep and Prune) for 2D and 3D to eliminate runtime branching.

Avoids costly conditional checks inside the innermost loops, ensuring optimal performance in 
high-frequency queries.

# Author: Louis D. 
# Created: 2023-04-20
# Python Version: 3.11
# Context: Need for a robust algorithm to detect broad-phase collisions between 2 or 
#          3-dimensional AABBs, reducing costly narrow-phase calculations in a Dynamic 
#          Relaxation solver.

"""

from numba import njit, prange, config
import numpy as np 


num_threads = config.NUMBA_NUM_THREADS


@njit("int32[:, :](int32[:], int32[:], int32[:], int32[:], int32[:])", parallel=True, fastmath=True, nogil=True, cache=True)
def SAP_2d(min_x, max_x, min_y, max_y, sorted_ids):

    """
    Optimized 2D Sweep and Prune (SAP) collision detection algorithm.

    Implements a cache-aware, multi-threaded broad-phase collision detector with:
    1. Contiguous sorted array ordering for improved memory locality and reduced cache misses
    2. Direct pairwise comparisons instead of vectorized boolean masks for better Numba performance
    3. Thread-local storage with parallel strided processing to minimize synchronization overhead
    4. Early pruning via sorted x-axis traversal to eliminate unnecessary intersection tests
    5. Dynamic buffer allocation based on problem size and thread count

    Parameters
    ----------
    min_x        (1d array): Minimum x-coordinates of bounding boxes
    max_x        (1d array): Maximum x-coordinates of bounding boxes
    min_y        (1d array): Minimum y-coordinates of bounding boxes
    max_y        (1d array): Maximum y-coordinates of bounding boxes
    sorted_ids   (1d array): Pre-sorted object indices by min_x coordinate

    Returns
    -------
    final_collisions (2d array): Array of collision pairs as [id1, id2] rows

    """

    n = min_x.shape[0]
    
    # Dynamic buffer allocation based on problem size and thread count
    buffer_size = max(1000, (n * n) // (num_threads * 20))  # Adjust based on expected collision density
    
    # Thread-local storage to eliminate synchronization overhead
    local_buffers = np.empty((num_threads, buffer_size, 2), dtype=np.int32)
    local_counts = np.zeros(num_threads, dtype=np.int32)
    
    # Preload sorted values for better cache locality and reduced memory indirection
    sorted_min_x = min_x[sorted_ids]
    sorted_max_x = max_x[sorted_ids]
    sorted_min_y = min_y[sorted_ids]
    sorted_max_y = max_y[sorted_ids]
    
    # Multi-threaded parallel processing with strided array access
    for thread_id in prange(num_threads):
        local_collisions = local_buffers[thread_id]
        local_count = 0
        
        # Each thread processes a stride of the array
        for i in range(thread_id, n, num_threads):
            current_min_x = sorted_min_x[i]
            current_max_x = sorted_max_x[i]
            current_min_y = sorted_min_y[i]
            current_max_y = sorted_max_y[i]
            
            # Process potential collisions
            for j in range(i + 1, n):
                # Early pruning for x-axis using sorted property to eliminate unnecessary checks
                if sorted_min_x[j] > current_max_x:
                    break
                
                # Direct comparison instead of boolean mask (faster in Numba)
                if (sorted_max_y[j] > current_min_y and sorted_min_y[j] < current_max_y):

                    # Safety check before adding to buffer
                    if local_count < buffer_size:
                        local_collisions[local_count, 0] = sorted_ids[i]
                        local_collisions[local_count, 1] = sorted_ids[j]
                        local_count += 1
            
        # Store count for this thread
        local_counts[thread_id] = local_count
    
    # Efficient merging of thread-local results into contiguous final array
    total_collisions = np.sum(local_counts)
    final_collisions = np.empty((total_collisions, 2), dtype=np.int32)
    
    idx = 0
    for thread_id in range(num_threads):
        count = local_counts[thread_id]
        if count > 0:
            final_collisions[idx:idx + count] = local_buffers[thread_id, :count]
            idx += count
    
    return final_collisions


@njit("int32[:, :](int32[:], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])", parallel=True, fastmath=True, nogil=True, cache=True)
def SAP_3d(min_x, max_x, min_y, max_y, min_z, max_z, sorted_ids):

    """
    Optimized 3D Sweep and Prune (SAP) collision detection algorithm.

    Parameters
    ----------
    min_x        (1d array): Minimum x-coordinates of bounding boxes
    max_x        (1d array): Maximum x-coordinates of bounding boxes
    min_y        (1d array): Minimum y-coordinates of bounding boxes
    max_y        (1d array): Maximum y-coordinates of bounding boxes
    min_z        (1d array): Minimum z-coordinates of bounding boxes
    max_z        (1d array): Maximum z-coordinates of bounding boxes
    sorted_ids   (1d array): Pre-sorted object indices by min_x coordinate

    Returns
    -------
    final_collisions (2d array): Array of collision pairs as [id1, id2] rows
    
    """

    n = min_x.shape[0]
    
    # Calculate buffer size more conservatively
    buffer_size = max(1000, (n * n) // (num_threads * 20))  # Adjust based on expected collision density
    
    # Thread-local storage
    local_buffers = np.empty((num_threads, buffer_size, 2), dtype=np.int32)
    local_counts = np.zeros(num_threads, dtype=np.int32)
    
    # Preload values for better cache locality
    sorted_min_x = min_x[sorted_ids]
    sorted_max_x = max_x[sorted_ids]
    sorted_min_y = min_y[sorted_ids]
    sorted_max_y = max_y[sorted_ids]
    sorted_min_z = min_z[sorted_ids]
    sorted_max_z = max_z[sorted_ids]
    
    # Parallel processing
    for thread_id in prange(num_threads):
        local_collisions = local_buffers[thread_id]
        local_count = 0
        
        # Each thread processes a stride of the array
        for i in range(thread_id, n, num_threads):
            current_min_x = sorted_min_x[i]
            current_max_x = sorted_max_x[i]
            current_min_y = sorted_min_y[i]
            current_max_y = sorted_max_y[i]
            current_min_z = sorted_min_z[i]
            current_max_z = sorted_max_z[i]
            
            # Process potential collisions
            for j in range(i + 1, n):
                # Early exit if beyond x-range (pruning)
                if sorted_min_x[j] > current_max_x:
                    break
                
                # Direct comparison instead of boolean mask (faster in Numba)
                if (sorted_max_y[j] > current_min_y and 
                    sorted_min_y[j] < current_max_y and
                    sorted_max_z[j] > current_min_z and 
                    sorted_min_z[j] < current_max_z):

                    # Safety check before adding to buffer
                    if local_count < buffer_size:
                        local_collisions[local_count, 0] = sorted_ids[i]
                        local_collisions[local_count, 1] = sorted_ids[j]
                        local_count += 1
            
        # Store count for this thread
        local_counts[thread_id] = local_count
    
    # Merge results
    total_collisions = np.sum(local_counts)
    final_collisions = np.empty((total_collisions, 2), dtype=np.int32)
    
    idx = 0
    for thread_id in range(num_threads):
        count = local_counts[thread_id]
        if count > 0:
            final_collisions[idx:idx + count] = local_buffers[thread_id, :count]
            idx += count
    
    return final_collisions


def query_pairs_3d(min_x, max_x, min_y, max_y, min_z, max_z):

    """
    Find all intersecting pairs of 3D axis-aligned bounding boxes.
    
    Uses a Sweep and Prune (SAP) algorithm with optimized memory layout
    and multi-threaded processing for efficient collision detection.
    
    Parameters
    ----------
    min_x       (1d array): Minimum x-coordinates of bounding boxes
    max_x       (1d array): Maximum x-coordinates of bounding boxes
    min_y       (1d array): Minimum y-coordinates of bounding boxes
    max_y       (1d array): Maximum y-coordinates of bounding boxes
    min_z       (1d array): Minimum z-coordinates of bounding boxes
    max_z       (1d array): Maximum z-coordinates of bounding boxes
    
    Returns
    -------
    pairs       (2d array): Array of collision pairs as [id1, id2] rows

    """

    # Sort indices
    sorted_ids = np.argsort(min_x).astype(np.int32)

    # Query pairs
    raw_pairs = SAP_3d(min_x, max_x, min_y, max_y, min_z, max_z, sorted_ids)

    if raw_pairs.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    return raw_pairs


def query_pairs_2d(min_x, max_x, min_y, max_y):

    """
    Find all intersecting pairs of 2D axis-aligned bounding boxes.
    
    Uses a Sweep and Prune (SAP) algorithm with optimized memory layout
    and multi-threaded processing for efficient collision detection.
    
    Parameters
    ----------
    min_x       (1d array): Minimum x-coordinates of bounding boxes
    max_x       (1d array): Maximum x-coordinates of bounding boxes
    min_y       (1d array): Minimum y-coordinates of bounding boxes
    max_y       (1d array): Maximum y-coordinates of bounding boxes
    
    Returns
    -------
    pairs       (2d array): Array of collision pairs as [id1, id2] rows

    """

    # Sort indices
    sorted_ids = np.argsort(min_x).astype(np.int32)

    # Query pairs
    raw_pairs = SAP_2d(min_x, max_x, min_y, max_y, sorted_ids)

    if raw_pairs.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    return raw_pairs