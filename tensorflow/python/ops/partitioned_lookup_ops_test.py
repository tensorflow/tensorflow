import tensorflow as tf
from tensorflow.python.framework import ops

import partitioned_lookup_ops

table = partitioned_lookup_ops.PartitionedMutableHashTable(tf.int64,
               tf.float32,
               [0, 0, 0],
               shard_num=2,
               name="PartitionedMutableHashTable",
               checkpoint=True)

keys = ops.convert_to_tensor([18287374, 7174746], dtype=tf.int64)
values = ops.convert_to_tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=tf.float32)
#out = table.contain(keys)
out = table.lookup(keys)
#out = table.size(keys)
op_list = table.insert(keys, values)

sess = tf.Session()
sess.run(op_list)
print(out.eval(session=sess))