import tensorflow as tf
from tensorflow.contrib.ignite import IgniteDataset

ds = IgniteDataset(
  cache_name="SQL_PUBLIC_TEST_CACHE", 
  port=42300
)
it = ds.make_one_shot_iterator()
ne = it.get_next()

with tf.Session() as sess:
  res = sess.run(ne)
  print(res)
