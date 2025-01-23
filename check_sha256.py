import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
nonces
def mine_chunk(start_nonce, chunk_size, target_zeros, thread_id):
    target = '0' * target_zeros
    end_nonce = start_nonce + chunk_size
    
    for nonce in range(start_nonce, end_nonce):
        message = f"block{nonce}{thread_id}".encode()
        current_hash = hashlib.sha256(message).hexdigest()
        if current_hash.startswith(target):
            return nonce, current_hash, thread_id
    return None, None, None

def mine_sha256_threaded(target_zeros, num_threads=8, chunk_size=1000000):
    start_nonce = 10692000000
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for nonce in nonces:
            for thread_id in range(num_threads):
                future = executor.submit(mine_chunk, start_nonce, chunk_size, target_zeros, thread_id)
                futures.append(future)
                start_nonce =
            
            for future in as_completed(futures):
                nonce, hash_result, thread_id = future.result()
                if nonce:
                    end_time = time.time()
                    print(f"Thread {thread_id} found matching hash!")
                    print(f"Nonce: {nonce}")
                    print(f"Message: block{nonce}{thread_id}")
                    print(f"Hash: {hash_result}")
                    print(f"Time taken: {end_time - start_time:.2f} seconds")
                    return nonce
            futures = []

if __name__ == "__main__":
    mine_sha256_threaded(8)