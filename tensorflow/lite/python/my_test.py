# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for mytest package."""


import logging
import traceback
import unittest

# pylint: disable=g-import-not-at-top
try:
  import tensorflow as tf
except ImportError as e:
  logging.exception("Error during import: %s", e)
  logging.exception("\nImport stack trace:")
  traceback.print_exc()
  raise  # Re-raise the exception so the test runner knows it failed
except Exception as e:
  logging.exception("A different error occurred during import: %s", e)
  logging.exception("\nImport stack trace:")
  traceback.print_exc()
  raise

try:
  from tensorflow.lite.python import lite
except ImportError as e:
  logging.exception("Error during import: %s", e)
  logging.exception("\nImport stack trace:")
  traceback.print_exc()
  raise  # Re-raise the exception so the test runner knows it failed
except Exception as e:
  logging.exception("A different error occurred during import: %s", e)
  logging.exception("\nImport stack trace:")
  traceback.print_exc()
  raise
# pylint: enable=g-import-not-at-top


class TestArithmeticFunctions(unittest.TestCase):
  def test_add_positive_numbers(self):
    logging.info("hello")
    logging.info(tf.__version__)
    logging.info(lite)

if __name__ == "__main__":
  unittest.main()
