# =============================================================================
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
# =============================================================================

# Python3 preparedness imports.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.proto.python.kernel_tests import test_case
from tensorflow.contrib.proto.python.ops import decode_proto_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class DecodeProtoFailTest(test_case.ProtoOpTestCase):
  """Test failure cases for DecodeToProto."""

  def _TestCorruptProtobuf(self, sanitize):
    """Test failure cases for DecodeToProto."""

    # The goal here is to check the error reporting.
    # Testing against a variety of corrupt protobufs is
    # done by fuzzing.
    corrupt_proto = 'This is not a binary protobuf'

    # Numpy silently truncates the strings if you don't specify dtype=object.
    batch = np.array(corrupt_proto, dtype=object)
    msg_type = 'tensorflow.contrib.proto.TestCase'
    field_names = ['sizes']
    field_types = [dtypes.int32]

    with self.test_session() as sess:
      ctensor, vtensor = decode_proto_op.decode_proto(
          batch,
          message_type=msg_type,
          field_names=field_names,
          output_types=field_types,
          sanitize=sanitize)
      with self.assertRaisesRegexp(errors.DataLossError,
                                   'Unable to parse binary protobuf'
                                   '|Failed to consume entire buffer'):
        _ = sess.run([ctensor] + vtensor)

  def testCorrupt(self):
    self._TestCorruptProtobuf(sanitize=False)

  def testSanitizerCorrupt(self):
    self._TestCorruptProtobuf(sanitize=True)


if __name__ == '__main__':
  test.main()
