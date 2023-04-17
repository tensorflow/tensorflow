# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of LoadDataset in Python."""
import multiprocessing
import os

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
# TODO(b/238903802): Use TypeSpec serialization methods directly.
from tensorflow.python.saved_model import nested_structure_coder


def _load(path, element_spec, compression, reader_func):
  return _LoadDataset(path, element_spec, compression, reader_func)


class _LoadDataset(dataset_ops.DatasetSource):
  """A dataset that loads previously saved dataset."""

  def __init__(self, path, element_spec=None, compression=None,
               reader_func=None):
    if reader_func is None:
      reader_func = lambda datasets: datasets.interleave(  # pylint:disable=g-long-lambda
          lambda x: x,
          cycle_length=multiprocessing.cpu_count(),
          num_parallel_calls=dataset_ops.AUTOTUNE)

    self._path = path
    if element_spec is None:
      with gfile.GFile(
          os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME), "rb") as f:
        encoded_spec = f.read()
      struct_pb = nested_structure_coder.struct_pb2.StructuredValue()
      struct_pb.ParseFromString(encoded_spec)
      spec = nested_structure_coder.decode_proto(struct_pb)
      self._element_spec = spec
    else:
      self._element_spec = element_spec
    self._compression = compression
    self._reader_func = structured_function.StructuredFunctionWrapper(
        reader_func,
        "load()",
        # Dataset of datasets of input elements
        input_structure=dataset_ops.DatasetSpec(
            dataset_ops.DatasetSpec(self._element_spec)))

    variant_tensor = ged_ops.load_dataset(
        path,
        reader_func_other_args=self._reader_func.function.captured_inputs,
        compression=compression,
        reader_func=self._reader_func.function,
        **self._flat_structure)
    super().__init__(variant_tensor)

  def _functions(self):
    return [self._reader_func]

  @property
  def element_spec(self):
    return self._element_spec
