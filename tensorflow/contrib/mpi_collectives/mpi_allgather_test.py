from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.mpi_collectives as mpi
from tensorflow.python.platform import test


average_allgather = False


class AllgatherTest(test.TestCase):
  def checkAllgather(self, num_ranks, all_gathered, local_gathered):
    # Ensure that indices match.
    all_gat_ind = np.sort(all_gathered.indices)
    loc_gat_ind = np.sort(local_gathered.indices)
    assert(len(loc_gat_ind) == len(all_gat_ind))
    for i in range(len(loc_gat_ind)):
      assert(loc_gat_ind[i] == all_gat_ind[i])

    # For each index, verify same values.
    local_checked = []
    for i in range(len(local_gathered.indices)):
      local_checked.append(False)
    for i in range(len(all_gathered.indices)):
      all_index = all_gathered.indices[i]
      # TODO(jthestness): Make this lookup quicker using sorting.
      loc_index = -1
      for j in range(len(local_gathered.indices)):
        if local_gathered.indices[j] == all_index and not local_checked[j]:
          loc_index = j
          local_checked[j] = True
          break
      assert(loc_index >= 0)
      correct_output = local_gathered.values[loc_index][0]
      if average_allgather:
        correct_output = correct_output / float(num_ranks)
      assert(all_gathered.values[i][0] == correct_output)


  def test_mpi_allgather(self):
    # Get MPI rank
    my_rank = int(os.environ['PMI_RANK'])
    num_ranks = int(os.environ['PMI_SIZE'])

    indices_per_rank = 100
    tensor_width = 10

    # Create IndexedSlices for each rank, some with overlapping indices.
    to_gather_indices = []
    to_gather_values = []
    to_gather = []
    for rank_id in range(num_ranks):
      indices = []
      values = []
      my_multiple = rank_id + 1
      current_index = my_multiple
      for i in range(indices_per_rank):
        indices.append(current_index)
        ones_tensor = tf.ones([tensor_width])
        values.append(tf.multiply(ones_tensor,
                                  tf.fill(ones_tensor.get_shape(),
                                          float(current_index))))
        current_index += my_multiple
      concat_ind = tf.stack(indices)
      concat_vals = tf.stack(values)
      to_gather_indices.append(concat_ind)
      to_gather_values.append(concat_vals)
      to_gather.append(tf.IndexedSlices(concat_vals, concat_ind))

    # Collect the local IndexedSlices (indices and values) to create
    # correct IndexedSlices output.
    correct_gather_indices = tf.concat(to_gather_indices, 0)
    correct_gather_values = tf.concat(to_gather_values, 0)
    correct_gather = tf.IndexedSlices(correct_gather_values,
                                      correct_gather_indices)

    all_gather = mpi.allreduce(to_gather[my_rank], average_allgather)

    # NOTE: This assumes that device IDs are numbered the same as ranks.
    gpu_options = tf.GPUOptions(visible_device_list=str(my_rank))
    config = tf.ConfigProto(gpu_options=gpu_options)

    # MPI Session to test allgather.
    with mpi.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())

      all_gathered, local_gathered = sess.run([all_gather, correct_gather])

      # Compare all_gathered with local_gathered.
      self.checkAllgather(num_ranks, all_gathered, local_gathered)


if __name__ == '__main__':
  test.main()
