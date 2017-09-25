from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.mpi_collectives as mpi
from tensorflow.python.platform import test


average_allreduce = False
max_wrong_count = -1


class AllreduceTest(test.TestCase):
  def dumpFailure(self, my_rank, out_loc_red, my_correct, out_all_red,
                  our_correct):
    # Find reduced/allreduced indices that are wrong and print all the
    # values from output, slices, reduced, allreduced, so we can debug
    # which is incorrect:
    wrong_count = 0
    red_dims = out_loc_red.shape
    assert(len(red_dims) == 2)
    for i in range(red_dims[0]):
      for j in range(red_dims[1]):
        suffix = ""
        if out_loc_red[i][j] != my_correct[i][j] or \
           out_all_red[i][j] != our_correct[i][j]:
          suffix = "WRONG"
          wrong_count += 1
        print("{}\t{}\t{}\t{}\t{}\t{}"
              .format(my_rank, i, j, out_loc_red[i][j],
                      out_all_red[i][j], suffix), flush=True)
        if max_wrong_count > 0 and wrong_count >= max_wrong_count:
          return

  def test_mpi_allreduce(self):
    # Get MPI rank
    my_rank = int(os.environ['PMI_RANK'])
    num_ranks = int(os.environ['PMI_SIZE'])

    stages = 13
    batch_size = 1331
    hidden_size = batch_size
    out_size = batch_size

    # Input placeholder (batch_size x hidden) - init to 1s
    inputs = tf.placeholder(tf.float32, shape=(batch_size, hidden_size),
                            name="Input")

    # Large matrices (hidden x out_dim) - init random
    weights = []
    for i in range(stages):
      initer = tf.constant_initializer(pow(2.0, i + 1.0))
      weights.append(tf.get_variable("weights_{}".format(i),
                                     shape=(hidden_size, out_size),
                                     dtype=tf.float32,
                                     initializer=initer))

    # Calculate output through dependent allreduces
    stage_input = inputs
    for i in range(stages):
      inter_output = tf.add(stage_input, weights[i],
                            name="add_red_{}".format(i))
      stage_input = mpi.allreduce(inter_output,
                                  average=average_allreduce)

    all_reduced = stage_input

    # Local reduced output for verification
    local_input = inputs
    for i in range(stages):
      inter_output = tf.add(local_input, weights[i],
                            name="addin_loc_{}".format(i))
      my_reducer = tf.Variable(initial_value=np.ones((hidden_size, out_size)),
                               dtype=tf.float32, name="loc_redr_{}".format(i))
      for r in range(num_ranks):
        my_reducer = tf.add(my_reducer, inter_output,
                            name="add_loc_{}_{}".format(i, r))
      if average_allreduce:
        local_input = tf.div(my_reducer, num_ranks,
                             name="div_loc_{}".format(i))
      else:
        local_input = my_reducer

    local_reduced = local_input

    # NOTE: This assumes that device IDs are numbered the same as ranks
    gpu_options = tf.GPUOptions(visible_device_list=str(my_rank))
    config = tf.ConfigProto(gpu_options=gpu_options)

    # MPI Session to test allreduce
    with mpi.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      input_feed = np.ones((batch_size, hidden_size), dtype=np.float32)
      our_output = input_feed[0][0]
      spread_var = 100
      input_feed = input_feed + my_rank * spread_var
      my_output = input_feed[0][0]
      for i in range(stages):
        curr_feed = my_output + pow(2.0, i + 1.0)
        my_output = curr_feed * num_ranks + 1
        curr_our_feed = our_output + pow(2.0, i + 1.0)
        if i == 0:
          sum_ranks = num_ranks * (num_ranks - 1) / 2
          our_output = curr_our_feed * num_ranks + \
            spread_var * sum_ranks
        else:
          our_output = curr_our_feed * num_ranks

      print("rank {}: My output is {}".format(my_rank, my_output))
      my_correct = np.zeros((batch_size, hidden_size), dtype=np.float32)
      my_correct = my_correct + my_output
      print("rank {}: Our output is {}".format(my_rank, our_output))
      our_correct = np.zeros((batch_size, hidden_size), dtype=np.float32)
      our_correct = our_correct + our_output

      for i in range(1000):
        if i % 100 == 0:
          print("{}: iter {}".format(my_rank, i), flush=True)
        feed_dict = {inputs: input_feed}
        out_all_red, out_loc_red \
          = sess.run([all_reduced, local_reduced],
                     feed_dict=feed_dict)

        if not np.allclose(out_loc_red, my_correct) or \
           not np.allclose(out_all_red, our_correct):
          print("Test incorrect on iter {}".format(i), flush=True)
          self.dumpFailure(my_rank, out_loc_red, my_correct, out_all_red,
                           our_correct)
          assert(np.allclose(out_loc_red, my_correct) and
                 np.allclose(out_all_red, our_correct))


if __name__ == '__main__':
  test.main()
