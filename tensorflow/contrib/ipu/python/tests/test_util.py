# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops


def create_multi_increasing_dataset(value,
                                    shapes=[[1, 32, 32, 4], [1, 8]],
                                    dtypes=[np.float32, np.float32],
                                    repeat=True):
  def _get_one_input(data):
    result = []
    for i in range(len(shapes)):
      result.append(
          math_ops.cast(
              gen_array_ops.broadcast_to(data, shape=shapes[i]),
              dtype=dtypes[i]))
    return result

  dataset = Dataset.range(value).map(_get_one_input)
  if repeat:
    dataset = dataset.repeat()
  return dataset


def create_dual_increasing_dataset(value,
                                   data_shape=[1, 32, 32, 4],
                                   label_shape=[1, 8],
                                   dtype=np.float32,
                                   repeat=True):
  return create_multi_increasing_dataset(
      value,
      shapes=[data_shape, label_shape],
      dtypes=[dtype, dtype],
      repeat=repeat)


def create_single_increasing_dataset(value,
                                     shape=[1, 32, 32, 4],
                                     dtype=np.float32,
                                     repeat=True):
  return create_multi_increasing_dataset(
      value, shapes=[shape], dtypes=[dtype], repeat=repeat)
