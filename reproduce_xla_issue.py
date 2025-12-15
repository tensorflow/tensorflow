import tensorflow as tf

@tf.function(jit_compile=True)
def mixed_dtypes_cond(pred):
  def true_fn():
    return tf.constant(1.0, dtype=tf.float32)
  def false_fn():
    return tf.constant(0, dtype=tf.int32)
  return tf.cond(pred, true_fn, false_fn)

try:
  print(mixed_dtypes_cond(tf.constant(True)))
except Exception as e:
  print(f"Caught exception: {e}")
