# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
import os

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.util import convert
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.avro.ops.gen_avro_record_dataset import avro_record_dataset

# Load the native method
this_dir = os.path.dirname(os.path.abspath(__file__))
lib_name = os.path.join(this_dir, '_avro_record_dataset.so')
reader_module = load_library.load_op_library(lib_name)

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB


class AvroRecordDataset(Dataset):
    """A `Dataset` comprising records from one or more Avro files."""

    def __init__(self, filenames, schema=None, buffer_size=None):
        """Creates a `AvroRecordDataset`.
        Args:
          filenames: A `tf.string` tensor containing one or more filenames.
          schema: (Optional.) A `tf.string` scalar for schema resolution.
          buffer_size: (Optional.) A `tf.int64` scalar representing the number of
            bytes in the read buffer. Must be larger >= 256.
        """
        super(AvroRecordDataset, self).__init__()

        # Force the type to string even if filenames is an empty list.
        self._filenames = ops.convert_to_tensor(filenames, dtypes.string, name="filenames")
        self._schema = convert.optional_param_to_tensor(
            "schema",
            schema,
            argument_default="",
            argument_dtype=dtypes.string)
        self._buffer_size = convert.optional_param_to_tensor(
            "buffer_size",
            buffer_size,
            argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)

    def _as_variant_tensor(self):
        return avro_record_dataset(self._filenames, self._schema,
                                   self._buffer_size)

    @property
    def output_classes(self):
        return ops.Tensor

    @property
    def output_shapes(self):
        return tensor_shape.TensorShape([])

    @property
    def output_types(self):
        return dtypes.string
