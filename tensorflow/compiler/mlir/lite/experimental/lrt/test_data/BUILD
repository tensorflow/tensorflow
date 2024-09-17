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

OUT_DIR = "$(RULEDIR)"

CONVERTER = "//tensorflow/compiler/mlir/lite:tf_tfl_translate"

CMD = """
for mlir_file in $(SRCS); do
    $(location {converter}) --input-mlir $$mlir_file --o={out_dir}/$$(basename $$mlir_file .mlir).tflite
done
""".format(
    converter = CONVERTER,
    out_dir = OUT_DIR,
)

genrule(
    name = "tflite_test_data",
    srcs = glob(["*.mlir"]),
    outs = [s.removesuffix(".mlir") + ".tflite" for s in glob(["*.mlir"])],
    cmd = CMD,
    tools = ["//tensorflow/compiler/mlir/lite:tf_tfl_translate"],
)

cc_library(
    name = "test_data_util",
    testonly = 1,
    hdrs = ["test_data_util.h"],
    data = [":tflite_test_data"],
    deps = [
        "//tensorflow/compiler/mlir/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:lite_rt_model_init",
        "//tensorflow/compiler/mlir/lite/experimental/lrt/core:model",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform",
    ],
)
