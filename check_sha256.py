import time
import hashlib
def mine_sha256(target_zeros = 5):
    nonce = 0
    target = '0' * target_zeros
    
    start_time = time.time()
    while True:
        message = f"George{nonce}".encode()
        current_hash = hashlib.sha256(message).hexdigest()
        
        if current_hash.startswith(target):
            end_time = time.time()
            print(f"Found matching hash after {nonce} attempts!")
            print(f"Nonce: {nonce}")
            print(f"Hash: {current_hash}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            break
            
        nonce += 1
    return nonce
mine_sha256(7)
input()