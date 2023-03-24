# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for SavedModel fingerprinting.

This module contains utility classes and functions for working with the
SavedModel fingerprint.
"""

from absl import logging

from tensorflow.core.config import flags
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import compat


def write_fingerprint(export_dir, saved_model_serialized):
  """Write fingerprint protobuf, if requested.

  Writes a `tf.saved_model.experimental.Fingerprint` object to a
  `fingerprint.pb` file in the `export_dir`.

  Args:
    export_dir: The directory in which to write the fingerprint.
    saved_model_serialized: The serialized SavedModel proto.
  """

  if flags.config().saved_model_fingerprinting.value():
    fingerprint_path = file_io.join(
        compat.as_str(export_dir),
        compat.as_str(constants.FINGERPRINT_FILENAME))
    logging.info("Writing fingerprint to %s", fingerprint_path)
    fingerprint_serialized = fingerprinting_pywrap.CreateFingerprintDef(
        saved_model_serialized, export_dir)
    file_io.atomic_write_string_to_file(fingerprint_path,
                                        fingerprint_serialized)
    # We need to deserialize the fingerprint in order to send its values.
    fingerprint_proto = fingerprint_pb2.FingerprintDef()
    fingerprint_proto.ParseFromString(fingerprint_serialized)
    metrics.SetWriteFingerprint(fingerprint=fingerprint_serialized)
    fingerprint = fingerprinting.Fingerprint.from_proto(fingerprint_serialized)
    metrics.SetWritePathAndSingleprint(path=export_dir,
                                       singleprint=fingerprint.singleprint())


def to_proto(fingerprint):
  if not isinstance(fingerprint, fingerprinting.Fingerprint):
    raise TypeError("Supplied value is not a Fingerprint.")
  return fingerprint_pb2.FingerprintDef(
      saved_model_checksum=fingerprint.saved_model_checksum,
      graph_def_program_hash=fingerprint.graph_def_program_hash,
      signature_def_hash=fingerprint.signature_def_hash,
      saved_object_graph_hash=fingerprint.saved_object_graph_hash,
      checkpoint_hash=fingerprint.checkpoint_hash)
