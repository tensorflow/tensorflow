# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for EncodeBase64 and DecodeBase64."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


@test_util.run_v1_only("b/120545219")
class Base64OpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._msg = array_ops.placeholder(dtype=dtypes.string)
    self._encoded_f = string_ops.encode_base64(self._msg, pad=False)
    self._decoded_f = string_ops.decode_base64(self._encoded_f)
    self._encoded_t = string_ops.encode_base64(self._msg, pad=True)
    self._decoded_t = string_ops.decode_base64(self._encoded_t)

  def _RemovePad(self, msg, base64_msg):
    if len(msg) % 3 == 1:
      return base64_msg[:-2]
    if len(msg) % 3 == 2:
      return base64_msg[:-1]
    return base64_msg

  def _RunTest(self, msg, pad):
    with self.cached_session() as sess:
      if pad:
        encoded, decoded = sess.run([self._encoded_t, self._decoded_t],
                                    feed_dict={self._msg: msg})
      else:
        encoded, decoded = sess.run([self._encoded_f, self._decoded_f],
                                    feed_dict={self._msg: msg})

    if not isinstance(msg, (list, tuple)):
      msg = [msg]
      encoded = [encoded]
      decoded = [decoded]

    base64_msg = [base64.urlsafe_b64encode(m) for m in msg]
    if not pad:
      base64_msg = [self._RemovePad(m, b) for m, b in zip(msg, base64_msg)]

    for i in range(len(msg)):
      self.assertEqual(base64_msg[i], encoded[i])
      self.assertEqual(msg[i], decoded[i])

  def testWithPythonBase64(self):
    for pad in (False, True):
      self._RunTest(b"", pad=pad)

      for _ in range(100):
        length = np.random.randint(1024 * 1024)
        msg = np.random.bytes(length)
        self._RunTest(msg, pad=pad)

  def testShape(self):
    for pad in (False, True):
      for _ in range(10):
        msg = [np.random.bytes(np.random.randint(20))
               for _ in range(np.random.randint(10))]
        self._RunTest(msg, pad=pad)

      # Zero-element, non-trivial shapes.
      for _ in range(10):
        k = np.random.randint(10)
        msg = np.empty((0, k), dtype=bytes)
        encoded = string_ops.encode_base64(msg, pad=pad)
        decoded = string_ops.decode_base64(encoded)

        with self.cached_session() as sess:
          encoded_value, decoded_value = self.evaluate([encoded, decoded])

        self.assertEqual(encoded_value.shape, msg.shape)
        self.assertEqual(decoded_value.shape, msg.shape)

  def testInvalidInput(self):
    def try_decode(enc):
      self._decoded_f.eval(feed_dict={self._encoded_f: enc})

    with self.cached_session():
      # Invalid length.
      msg = np.random.bytes(99)
      enc = base64.urlsafe_b64encode(msg)
      with self.assertRaisesRegexp(errors.InvalidArgumentError, "1 modulo 4"):
        try_decode(enc + b"a")

      # Invalid char used in encoding.
      msg = np.random.bytes(34)
      enc = base64.urlsafe_b64encode(msg)
      for i in range(len(msg)):
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"?" + enc[(i + 1):])
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"\x80" + enc[(i + 1):])  # outside ascii range.
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"+" + enc[(i + 1):])  # not url-safe.
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"/" + enc[(i + 1):])  # not url-safe.

      # Partial padding.
      msg = np.random.bytes(34)
      enc = base64.urlsafe_b64encode(msg)
      with self.assertRaises(errors.InvalidArgumentError):
        # enc contains == at the end. Partial padding is not allowed.
        try_decode(enc[:-1])

      # Unnecessary padding.
      msg = np.random.bytes(33)
      enc = base64.urlsafe_b64encode(msg)
      with self.assertRaises(errors.InvalidArgumentError):
        try_decode(enc + b"==")
      with self.assertRaises(errors.InvalidArgumentError):
        try_decode(enc + b"===")
      with self.assertRaises(errors.InvalidArgumentError):
        try_decode(enc + b"====")

      # Padding in the middle. (Previous implementation was ok with this as long
      # as padding char location was 2 or 3 (mod 4).
      msg = np.random.bytes(33)
      enc = base64.urlsafe_b64encode(msg)
      for i in range(len(msg) - 1):
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"=" + enc[(i + 1):])
      for i in range(len(msg) - 2):
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"==" + enc[(i + 2):])


if __name__ == "__main__":
  test.main()
