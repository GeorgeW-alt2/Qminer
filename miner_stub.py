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
 
proof_of_work_size = 10
range_compute = 1000000 #needs to be computable
_range =        1119000000 #simply add optimal ghost values to range then rerun
SHAmsg = "George"
message ="" #leave empty
ghost_values = [513621000000, 2039937000000, 2765049000000, 5456244000000, 6167928000000, 6844923000000, 7596891000000, 8322003000000, 9804678000000, 10526433000000, 12122127000000, 1259994000000, 12778980000000, 3410712000000, 4115682000000, 4816176000000, 11324280000000, 9132159000000]
ghost_values = shuffle_array(ghost_values)	
mine_sha256_threaded(proof_of_work_size, ghost_values, _range)
input()