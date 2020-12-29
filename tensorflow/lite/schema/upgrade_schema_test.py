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
"""Testing for updating TensorFlow lite schema."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tempfile
from tensorflow.lite.schema import upgrade_schema as upgrade_schema_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib

EMPTY_TEST_SCHEMA_V1 = {
    "version": 1,
    "operator_codes": [],
    "subgraphs": [],
}

EMPTY_TEST_SCHEMA_V3 = {
    "version": 3,
    "operator_codes": [],
    "subgraphs": [],
    "buffers": [{
        "data": []
    }]
}

TEST_SCHEMA_V0 = {
    "operator_codes": [],
    "tensors": [],
    "inputs": [],
    "outputs": [],
    "operators": [],
    "version": 0
}

TEST_SCHEMA_V3 = {
    "operator_codes": [],
    "buffers": [{
        "data": []
    }],
    "subgraphs": [{
        "tensors": [],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "version":
        3
}

FULL_TEST_SCHEMA_V1 = {
    "version":
        1,
    "operator_codes": [
        {
            "builtin_code": "CONVOLUTION"
        },
        {
            "builtin_code": "DEPTHWISE_CONVOLUTION"
        },
        {
            "builtin_code": "AVERAGE_POOL"
        },
        {
            "builtin_code": "MAX_POOL"
        },
        {
            "builtin_code": "L2_POOL"
        },
        {
            "builtin_code": "SIGMOID"
        },
        {
            "builtin_code": "L2NORM"
        },
        {
            "builtin_code": "LOCAL_RESPONSE_NORM"
        },
        {
            "builtin_code": "ADD"
        },
        {
            "builtin_code": "Basic_RNN"
        },
    ],
    "subgraphs": [{
        "operators": [
            {
                "builtin_options_type": "PoolOptions"
            },
            {
                "builtin_options_type": "DepthwiseConvolutionOptions"
            },
            {
                "builtin_options_type": "ConvolutionOptions"
            },
            {
                "builtin_options_type": "LocalResponseNormOptions"
            },
            {
                "builtin_options_type": "BasicRNNOptions"
            },
        ],
    }],
    "description":
        "",
}

FULL_TEST_SCHEMA_V3 = {
    "version":
        3,
    "operator_codes": [
        {
            "builtin_code": "CONV_2D"
        },
        {
            "builtin_code": "DEPTHWISE_CONV_2D"
        },
        {
            "builtin_code": "AVERAGE_POOL_2D"
        },
        {
            "builtin_code": "MAX_POOL_2D"
        },
        {
            "builtin_code": "L2_POOL_2D"
        },
        {
            "builtin_code": "LOGISTIC"
        },
        {
            "builtin_code": "L2_NORMALIZATION"
        },
        {
            "builtin_code": "LOCAL_RESPONSE_NORMALIZATION"
        },
        {
            "builtin_code": "ADD"
        },
        {
            "builtin_code": "RNN"
        },
    ],
    "subgraphs": [{
        "operators": [
            {
                "builtin_options_type": "Pool2DOptions"
            },
            {
                "builtin_options_type": "DepthwiseConv2DOptions"
            },
            {
                "builtin_options_type": "Conv2DOptions"
            },
            {
                "builtin_options_type": "LocalResponseNormalizationOptions"
            },
            {
                "builtin_options_type": "RNNOptions"
            },
        ],
    }],
    "description":
        "",
    "buffers": [{
        "data": []
    }]
}

BUFFER_TEST_V2 = {
    "operator_codes": [],
    "buffers": [],
    "subgraphs": [{
        "tensors": [
            {
                "data_buffer": [1, 2, 3, 4]
            },
            {
                "data_buffer": [1, 2, 3, 4, 5, 6, 7, 8]
            },
            {
                "data_buffer": []
            },
        ],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "version":
        2
}

BUFFER_TEST_V3 = {
    "operator_codes": [],
    "subgraphs": [{
        "tensors": [
            {
                "buffer": 1
            },
            {
                "buffer": 2
            },
            {
                "buffer": 0
            },
        ],
        "inputs": [],
        "outputs": [],
        "operators": [],
    }],
    "buffers": [
        {
            "data": []
        },
        {
            "data": [1, 2, 3, 4]
        },
        {
            "data": [1, 2, 3, 4, 5, 6, 7, 8]
        },
    ],
    "version":
        3
}


def JsonDumpAndFlush(data, fp):
  """Write the dictionary `data` to a JSON file `fp` (and flush).

  Args:
    data: in a dictionary that is JSON serializable.
    fp: File-like object
  """
  json.dump(data, fp)
  fp.flush()


class TestSchemaUpgrade(test_util.TensorFlowTestCase):

  def testNonExistentFile(self):
    converter = upgrade_schema_lib.Converter()
    non_existent = tempfile.mktemp(suffix=".json")
    with self.assertRaisesRegex(IOError, "No such file or directory"):
      converter.Convert(non_existent, non_existent)

  def testInvalidExtension(self):
    converter = upgrade_schema_lib.Converter()
    invalid_extension = tempfile.mktemp(suffix=".foo")
    with self.assertRaisesRegex(ValueError, "Invalid extension on input"):
      converter.Convert(invalid_extension, invalid_extension)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_json:
      JsonDumpAndFlush(EMPTY_TEST_SCHEMA_V1, in_json)
      with self.assertRaisesRegex(ValueError, "Invalid extension on output"):
        converter.Convert(in_json.name, invalid_extension)

  def CheckConversion(self, data_old, data_expected):
    """Given a data dictionary, test upgrading to current version.

    Args:
        data_old: TFLite model as a dictionary (arbitrary version).
        data_expected: TFLite model as a dictionary (upgraded).
    """
    converter = upgrade_schema_lib.Converter()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as in_json, \
            tempfile.NamedTemporaryFile(
                suffix=".json", mode="w+") as out_json, \
            tempfile.NamedTemporaryFile(
                suffix=".bin", mode="w+b") as out_bin, \
            tempfile.NamedTemporaryFile(
                suffix=".tflite", mode="w+b") as out_tflite:
      JsonDumpAndFlush(data_old, in_json)
      # Test JSON output
      converter.Convert(in_json.name, out_json.name)
      # Test binary output
      # Convert to .tflite  and then to .bin and check if binary is equal
      converter.Convert(in_json.name, out_tflite.name)
      converter.Convert(out_tflite.name, out_bin.name)
      self.assertEqual(
          open(out_bin.name, "rb").read(),
          open(out_tflite.name, "rb").read())
      # Test that conversion actually produced successful new json.
      converted_schema = json.load(out_json)
      self.assertEqual(converted_schema, data_expected)

  def testAlreadyUpgraded(self):
    """A file already at version 3 should stay at version 3."""
    self.CheckConversion(EMPTY_TEST_SCHEMA_V3, EMPTY_TEST_SCHEMA_V3)
    self.CheckConversion(TEST_SCHEMA_V3, TEST_SCHEMA_V3)
    self.CheckConversion(BUFFER_TEST_V3, BUFFER_TEST_V3)

  # Disable this while we have incorrectly versioned structures around.
  # def testV0Upgrade_IntroducesSubgraphs(self):
  #   """V0 did not have subgraphs; check to make sure they get introduced."""
  #   self.CheckConversion(TEST_SCHEMA_V0, TEST_SCHEMA_V3)

  def testV1Upgrade_RenameOps(self):
    """V1 had many different names for ops; check to make sure they rename."""
    self.CheckConversion(EMPTY_TEST_SCHEMA_V1, EMPTY_TEST_SCHEMA_V3)
    self.CheckConversion(FULL_TEST_SCHEMA_V1, FULL_TEST_SCHEMA_V3)

  def testV2Upgrade_CreateBuffers(self):
    """V2 did not have buffers; check to make sure they are created."""
    self.CheckConversion(BUFFER_TEST_V2, BUFFER_TEST_V3)


if __name__ == "__main__":
  test_lib.main()
