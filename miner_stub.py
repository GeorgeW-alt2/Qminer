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
 
proof_of_work_size = 8
range_compute = 100000 #needs to be computable
_range =        1119000000 #simply add optimal ghost values to range then rerun
SHAmsg = "GeorgeW"
message ="" #leave empty
ghost_values = [15845088000000, 15038576000000, 15004528000000, 11997664000000, 13530888000000, 14203336000000, 9736664000000, 10460184000000, 12761616000000, 8258768000000, 449008000000, 2421664000000, 4302816000000, 5576424000000, 6213760000000, 7586320000000, 3714424000000, 1781136000000, 11226264000000, 8991864000000, 1108688000000, 3076024000000, 4929512000000, 6895784000000]
ghost_values = shuffle_array(ghost_values)	
mine_sha256_threaded(proof_of_work_size, ghost_values, range_compute)
input()