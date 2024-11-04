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
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

# TODO: b/365295276 - Make custom rule and move to `.sh`.
OUT_DIR = "$(RULEDIR)"

CONVERTER = "//tensorflow/compiler/mlir/lite:tf_tfl_translate"

CMD = """
for mlir_file in $(SRCS); do
    $(location {converter}) --input-mlir $$mlir_file --o={out_dir}/testdata/$$(basename $$mlir_file .mlir).tflite
done
""".format(
    converter = CONVERTER,
    out_dir = OUT_DIR,
)

genrule(
    name = "tflite_test_data",
    srcs = glob(["testdata/*.mlir"]),
    outs = [s.removesuffix(".mlir") + ".tflite" for s in glob(["testdata/*.mlir"])],
    cmd = CMD,
    tools = [CONVERTER],
)

cc_library(
    name = "common",
    testonly = 1,
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
        "//tensorflow/lite/experimental/litert/cc:litert_cc_api",
        "//tensorflow/lite/experimental/litert/core:litert_model_init",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@flatbuffers//:runtime_cc",
        "@local_tsl//tsl/platform",
    ],
)

cc_library(
    name = "simple_model",
    testonly = 1,
    hdrs = [
        "testdata/simple_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_model.tflite",
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
    ],
)

cc_library(
    name = "simple_model_npu",
    testonly = 1,
    srcs = [],
    hdrs = [
        "testdata/simple_model_test_vectors.h",
    ],
    data = [
        "testdata/simple_model_google_tensor.bin",
        "testdata/simple_model_npu.tflite",
        "testdata/simple_model_qualcomm.bin",
    ],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_c_api",
    ],
)
