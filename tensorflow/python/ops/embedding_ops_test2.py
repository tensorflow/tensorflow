import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.lookup import lookup_ops

import partitioned_lookup_ops
import embedding_ops

init_op_list = []

emb_table = partitioned_lookup_ops.PartitionedMutableHashTable(tf.int64,
                                                               tf.float32,
                                                               [0.0, 0.0, 0.0],
                                                               shard_num=1,
                                                               name="sparse_id_embedding",
                                                               checkpoint=True)
keys = ops.convert_to_tensor([18287374, 7174746], dtype=tf.int64)
values = ops.convert_to_tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=tf.float32)
init_op_list.append(emb_table.insert(keys, values))

#count_table = partitioned_lookup_ops.PartitionedMutableHashTable(tf.int64,
#                                                                 tf.int64,
#                                                                 0,
#                                                                 shard_num=5,
#                                                                 name="sparse_id_counter",
#                                                                 checkpoint=True)
#keys = ops.convert_to_tensor([18287374, 7174746], dtype=tf.int64)
#values = ops.convert_to_tensor([1, 1], dtype=tf.int64)
#init_op_list.append(count_table.insert(keys, values))

sp_ids = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1]],
                         values=[18287374, 3847113, 7174746, 18287374, 5173648, 5173648],
                         dense_shape=[4, 1000])
sp_weights = tf.SparseTensor(indices=[[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1]],
                             values=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             dense_shape=[4, 1000])
embedding = embedding_ops.embedding_lookup_sparse_with_hash_table(emb_table,
                                                                  sp_ids,
                                                                  sp_weights,
                                                                  combiner="sum",
                                                                  is_training=True,
                                                                  count_table=None,
                                                                  count_filter_thr=1)
weight = tf.get_variable("weight", initializer=tf.random_normal([3, 2], dtype=tf.float32))
logit = tf.matmul(embedding, weight)
label = ops.convert_to_tensor([1, 0, 1, 0], dtype=tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label)
loss = tf.reduce_mean(cross_entropy)

opt = tf.train.AdagradOptimizer(learning_rate=0.01)
grads = tf.gradients(loss, [embedding])
train = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.write_graph(sess.graph.as_graph_def(), './', 'graph.txt', as_text=True)
    writer = tf.summary.FileWriter(logdir='./logdir', graph=sess.graph)
    writer.flush()
    exit()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(init_op_list)
#print(embedding.eval(session=sess))

i = 0
while i < 3:
  sess.run(train)
  print("loss = %d" % loss.eval(session=sess))
  print("embedding = %s" % embedding.eval(session=sess))
  print("grads = %s" % grads[0].eval(session=sess))
  i += 1
