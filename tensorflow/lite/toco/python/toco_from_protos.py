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
"""Python console command to invoke TOCO from serialized protos."""
import argparse
import sys

# We need to import pywrap_tensorflow prior to the toco wrapper.
# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api
from absl import app

FLAGS = None


def execute(unused_args):
  """Runs the converter."""
  with open(FLAGS.model_proto_file, "rb") as model_file:
    model_str = model_file.read()

  with open(FLAGS.toco_proto_file, "rb") as toco_file:
    toco_str = toco_file.read()

  with open(FLAGS.model_input_file, "rb") as input_file:
    input_str = input_file.read()

  output_str = _pywrap_toco_api.TocoConvert(
      model_str,
      toco_str,
      input_str,
      False,  # extended_return
  )
  open(FLAGS.model_output_file, "wb").write(output_str)
  sys.exit(0)


def main():
  global FLAGS
  parser = argparse.ArgumentParser(
      description="Invoke toco using protos as input.")
  parser.add_argument(
      "model_proto_file",
      type=str,
      help="File containing serialized proto that describes the model.")
  parser.add_argument(
      "toco_proto_file",
      type=str,
      help="File containing serialized proto describing how TOCO should run.")
  parser.add_argument(
      "model_input_file", type=str, help="Input model is read from this file.")
  parser.add_argument(
      "model_output_file",
      type=str,
      help="Result of applying TOCO conversion is written here.")

  FLAGS, unparsed = parser.parse_known_args()

  app.run(main=execute, argv=[sys.argv[0]] + unparsed)


if __name__ == "__main__":
  main()
