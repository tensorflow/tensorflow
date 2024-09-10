# Copyright 2024 Google LLC.
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

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/mlir/lite/experimental/lrt:__subpackages__"],
)

# TODO: b/365295276 - Make custom rule and move to `.sh`.
IN_DIR = "third_party/tensorflow/compiler/mlir/lite/experimental/lrt/test_data"

OUT_DIR = "$(RULEDIR)"

CONVERTER = "//tensorflow/compiler/mlir/lite:tf_tfl_translate"

CMD = """
for mlir_file in {in_dir}/*.mlir; do
    $(location {converter}) --input-mlir $$mlir_file --o={out_dir}/$$(basename $$mlir_file .mlir).tflite
done
""".format(
    converter = CONVERTER,
    in_dir = IN_DIR,
    out_dir = OUT_DIR,
)

genrule(
    name = "tflite_test_data",
    srcs = glob(["*.mlir"]),
    outs = [s.removesuffix(".mlir") + ".tflite" for s in glob(["*.mlir"])],
    cmd = CMD,
    tools = ["//tensorflow/compiler/mlir/lite:tf_tfl_translate"],
)
