import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Testing complex Variable operations with tf.raw_ops.Conj and assign_add...")

try:
    input_data = tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)
    var = tf.Variable(input_data, dtype=tf.complex64)
    conj_result = tf.raw_ops.Conj(input=input_data)
    assign_add_op = var.assign_add(conj_result)
    print("Success! Result:", assign_add_op.numpy())
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
