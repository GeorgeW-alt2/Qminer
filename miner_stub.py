import random
import hashlib
def shuffle_array(arr):
    n = len(arr)
    for i in range(n-1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]
    return arr
def mine_chunk(start_nonce, chunk_size, target_zeros, thread_id):
    target = '0' * target_zeros
    end_nonce = start_nonce + chunk_size
    
    for nonce in range(start_nonce, end_nonce):
        message = f"{SHAmsg}{nonce}".encode()
        current_hash = hashlib.sha256(message).hexdigest()
        if current_hash.startswith(target):
            return nonce, current_hash, thread_id
    return None, None, None
    
def load_ghost_values(filename='ghost_values.csv'):
    ghost_values = []
    with open(filename, 'r') as f:
        reader = f.readlines()
        for row in reader:
            ghost_values.append(int(row))
    return ghost_values
    
def mine_sha256_threaded(target_zeros, ghost_values, chunk_size):
    
    for nonce in ghost_values:
        print(f"\nTrying ghost value: {nonce}")
        result = mine_chunk(nonce, chunk_size, proof_of_work_size, 0)
        
        if result[0]:
            nonce, hash_result, _ = result
            print(f"\nFound matching hash!")
            print(f"Nonce: {nonce}")
            print(f"message: {SHAmsg}{nonce}")
            print(f"Hash: {hash_result}")
            return nonce
 
proof_of_work_size = 9
range_compute = 200000000 #needs to be computable
_range =        100000000000 #simply add optimal ghost values to range then rerun
SHAmsg = "GeorgeW"
message ="" #leave empty
ghost_values = load_ghost_values()
ghost_values = shuffle_array(ghost_values)	
mine_sha256_threaded(proof_of_work_size, ghost_values, range_compute)
input()