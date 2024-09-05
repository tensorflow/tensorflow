# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

r"""Generates SavedModel test data for Merger.

Constructs chunked proto test data containing a SavedModel.

Example command:

bazel run tensorflow/tools/proto_splitter/testdata:split_saved_model_gen -- \
  --path /tmp \
  --saved_model_type=split-standard \
  --export=pb,cpb
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter.python import saved_model as split_saved_model
from tensorflow.tools.proto_splitter.python import test_util


STANDARD_SIZES = [100, 100, 1000, 100, 1000, 500, 100, 100, 100]


def _split_and_write(
    path: str,
    saved_model: saved_model_pb2.SavedModel,
    max_size: int,
    export_files: Sequence[str],
):
  """Writes the .pb, .pbtxt and .cpb files for a SavedModel."""
  constants.debug_set_max_size(max_size)

  if "pbtxt" in export_files:
    output_path = f"{path}.pbtxt"
    file_io.write_string_to_file(output_path, str(saved_model))
    logging.info("  %s written", output_path)
  if "pb" in export_files:
    output_path = f"{path}.pb"
    file_io.write_string_to_file(output_path, saved_model.SerializeToString())
    logging.info("  %s written", output_path)
  if "cpb" in export_files:
    splitter = split_saved_model.SavedModelSplitter(saved_model)
    splitter.write(path)
    chunks, _ = splitter.split()
    if len(chunks) > 1:
      logging.info("  %s.cpb written", path)
    else:
      raise RuntimeError(
          "For some reason this graph was not chunked, so a .cpb file was not"
          " produced. Raising an error since this should not be the case."
      )


def split_standard(path: str, export_files: Sequence[str]):
  """Splits a standard SavedModel."""
  fn1 = [100, 100, 100]
  fn2 = [100, 500]
  fn3 = [100]
  fn4 = [100, 100]

  max_size = 500
  constants.debug_set_max_size(max_size)

  graph_def = test_util.make_graph_def_with_constant_nodes(
      STANDARD_SIZES, fn1=fn1, fn2=fn2, fn3=fn3, fn4=fn4
  )
  proto = saved_model_pb2.SavedModel()
  proto.meta_graphs.add().graph_def.CopyFrom(graph_def)

  _split_and_write(path, proto, max_size, export_files)


VALID_SAVED_MODEL_TYPES = {
    "split-standard": split_standard,
}
ALL_SAVED_MODEL_TYPES = ", ".join(VALID_SAVED_MODEL_TYPES.keys())

SPLITTER_TESTDATA_PATH = flags.DEFINE_string(
    "path", None, help="Path to testdata directory."
)
SAVED_MODEL_TYPES = flags.DEFINE_multi_string(
    "saved_model_type",
    "all",
    help=(
        "Type(s) of saved model to export. Valid types: all, "
        f"{ALL_SAVED_MODEL_TYPES}"
    ),
)
EXPORT_FILES = flags.DEFINE_multi_string(
    "export",
    "all",
    help="List of files to export. Valid options: all, pb, pbtxt, cpb",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if "all" in EXPORT_FILES.value:
    export_files = ["pb", "pbtxt", "cpb"]
  else:
    export_files = EXPORT_FILES.value

  if "all" in SAVED_MODEL_TYPES.value:
    saved_model_types = VALID_SAVED_MODEL_TYPES.keys()
  else:
    saved_model_types = SAVED_MODEL_TYPES.value

  for v in saved_model_types:
    if v not in VALID_SAVED_MODEL_TYPES:
      raise ValueError(
          "Invalid flag passed to `saved_model_type`: "
          f"{v}\nValid saved model types:"
          f" {ALL_SAVED_MODEL_TYPES}"
      )

    logging.info("Generating saved model %s", v)
    f = VALID_SAVED_MODEL_TYPES[v]
    f(os.path.join(SPLITTER_TESTDATA_PATH.value, v), export_files)


if __name__ == "__main__":
  app.run(main)
