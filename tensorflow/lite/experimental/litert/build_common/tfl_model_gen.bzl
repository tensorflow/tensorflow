# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility to generate tflite models from MLIR files."""

def tfl_model_gen(name, srcs, subdir = "testdata"):
    """
    Generates tflite models from MLIR files.

    Args:
      name: name of the rule.
      srcs: list of MLIR files.
      subdir: subdirectory to place the generated tflite files.
    """
    OUT_DIR = "$(RULEDIR)"
    CONVERTER = "//tensorflow/compiler/mlir/lite:tf_tfl_translate"
    CMD = """
    for mlir_file in $(SRCS); do
        $(location {converter}) --input-mlir $$mlir_file --o={out_dir}/{subdir}/$$(basename $$mlir_file .mlir).tflite
    done
    """.format(
        converter = CONVERTER,
        out_dir = OUT_DIR,
        subdir = subdir,
    )

    native.genrule(
        name = name,
        srcs = srcs,
        outs = [s.removesuffix(".mlir") + ".tflite" for s in srcs],
        cmd = CMD,
        tools = [CONVERTER],
    )

    native.filegroup(
        name = name + "_files",
        srcs = [name],
    )
