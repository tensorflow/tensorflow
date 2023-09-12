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

FingerprintException = fingerprinting_pywrap.FingerprintException


def write_fingerprint(export_dir):
  """Write fingerprint protobuf, if requested.

  Writes a `tf.saved_model.experimental.Fingerprint` object to a
  `fingerprint.pb` file in the `export_dir` using the `saved_model.pb` file
  contained in `export_dir`.

  Args:
    export_dir: The directory in which to write the fingerprint.
  """
  if flags.config().saved_model_fingerprinting.value():
    fingerprint_path = file_io.join(
        compat.as_str(export_dir),
        compat.as_str(constants.FINGERPRINT_FILENAME))
    logging.info("Writing fingerprint to %s", fingerprint_path)

    try:
      fingerprint_serialized = fingerprinting_pywrap.CreateFingerprintDef(
          export_dir)
    except FingerprintException as e:
      raise ValueError(e) from None
    file_io.atomic_write_string_to_file(fingerprint_path,
                                        fingerprint_serialized)

    metrics.SetWriteFingerprint(fingerprint=fingerprint_serialized)
    try:
      metrics.SetWritePathAndSingleprint(
          path=export_dir,
          singleprint=singleprint_from_fingerprint_proto(export_dir))
    except metrics.MetricException:
      logging.info("path_and_singleprint metric could not be set. "
                   "Model saving will continue.")


def singleprint_from_saved_model_proto(export_dir):
  """Returns the singleprint of `saved_model.pb` in `export_dir`.

  Args:
    export_dir: The directory that contains `saved_model.pb`.

  Returns:
    A string containing the singleprint of `saved_model.pb` in `export_dir`.

  Raises:
    ValueError: If a valid singleprint cannot be constructed from 
    `saved_model.pb`.
  """
  try:
    return fingerprinting_pywrap.SingleprintFromSM(export_dir)
  except FingerprintException as e:
    raise ValueError(e) from None


def singleprint_from_fingerprint_proto(export_dir):
  """Returns the singleprint of `fingerprint.pb` in `export_dir`.

  Args:
    export_dir: The directory that contains `fingerprint.pb`.

  Returns:
    A string containing the singleprint of `fingerprint.pb` in `export_dir`.

  Raises:
    ValueError: If a valid singleprint cannot be constructed from
    `fingerprint.pb`.
  """
  try:
    return fingerprinting_pywrap.SingleprintFromFP(export_dir)
  except FingerprintException as e:
    raise ValueError(e) from None


def singleprint_from_saved_model(export_dir):
  """Returns the singleprint of the SavedModel in `export_dir`.

  First tries to construct the singleprint from `fingerprint.pb`, then from
  `saved_model.pb`. Attempts to write the `fingerprint.pb` if not found, but
  doesn't return an error if it isn't writeable.

  Args:
    export_dir: The directory that contains the SavedModel.

  Returns:
    A string containing the singleprint of the SavedModel in `export_dir`.

  Raises:
    ValueError: If a valid singleprint cannot be constructed from the
    SavedModel.
  """
  # try generating the singleprint from `fingerprint.pb`
  try:
    return singleprint_from_fingerprint_proto(export_dir)
  except FingerprintException:
    pass

  # try creating `fingerprint.pb`, then generating the singleprint
  try:
    write_fingerprint(export_dir)
    return singleprint_from_fingerprint_proto(export_dir)
  except FingerprintException:
    pass

  # try generating the singleprint from `saved_model.pb`
  try:
    return singleprint_from_saved_model_proto(export_dir)
  except FingerprintException as e:
    raise ValueError(e) from None


def to_proto(fingerprint):
  if not isinstance(fingerprint, fingerprinting.Fingerprint):
    raise TypeError("Supplied value is not a Fingerprint.")
  return fingerprint_pb2.FingerprintDef(
      saved_model_checksum=fingerprint.saved_model_checksum,
      graph_def_program_hash=fingerprint.graph_def_program_hash,
      signature_def_hash=fingerprint.signature_def_hash,
      saved_object_graph_hash=fingerprint.saved_object_graph_hash,
      checkpoint_hash=fingerprint.checkpoint_hash)
