import tensorflow as tf

# Define the tensors
A = tf.constant([[2, 2], [2, 2]], dtype=tf.int64)
B = tf.constant([[0, 0], [0, 0]], dtype=tf.int64)

def safe_floormod(A, B):
    if tf.reduce_any(B == 0):
        raise tf.errors.InvalidArgumentError(None, None, "Integer division by zero.")
    return tf.math.floormod(A, B)

# Using the wrapper function
try:
    output_gpu = safe_floormod(A, B)
    print(f"GPU output: {output_gpu}")
except tf.errors.InvalidArgumentError as e:
    print(f"Error on GPU: {e}")

# Attempt to perform floormod on GPU
try:
    with tf.device("/gpu:0"):
        output_gpu = tf.math.floormod(A, B)  # This will raise an error
        print(f"\nGPU output: {output_gpu}\n")
except tf.errors.InvalidArgumentError as e:
    print(f"Error on GPU: {e}")