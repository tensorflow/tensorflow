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

load("//tensorflow/lite/experimental/litert/build_common:tfl_model_gen.bzl", "tfl_model_gen")
load("//tensorflow/lite/experimental/litert/integration_test:run_on_device.bzl", "device_rlocation", "get_driver", "run_on_device")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

# C++ TEST SCAFFOLD ################################################################################

cc_test(
    name = "gen_device_test",
    srcs = ["gen_device_test.cc"],
    copts = ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    data = [":single_op_models"],
    linkopts = select({
        "//tensorflow:android": ["-landroid"],
        "//conditions:default": [],
    }),
    tags = ["manual"],
    deps = [
        ":gen_device_test_lib",
        "//tensorflow/lite/experimental/litert/c:litert_logging",
        "//tensorflow/lite/experimental/litert/cc:litert_environment",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "//tensorflow/lite/experimental/litert/tools:dump",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "gen_device_test_lib",
    testonly = True,
    hdrs = ["gen_device_test_lib.h"],
    deps = [
        "//tensorflow/lite/experimental/litert/c:litert_common",
        "//tensorflow/lite/experimental/litert/cc:litert_compiled_model",
        "//tensorflow/lite/experimental/litert/cc:litert_environment",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/cc:litert_tensor_buffer",
        "//tensorflow/lite/experimental/litert/core/model",
        "//tensorflow/lite/experimental/litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

# TEST MODELS ######################################################################################

filegroup(
    name = "classic_ml_models",
    srcs = glob(["classic_ml_models/*.tflite"]),
)

tfl_model_gen(
    name = "single_op_models",
    srcs = glob(["single_op_models/*.mlir"]),
    subdir = "single_op_models",
)

filegroup(
    name = "pre_compiled_models",
    srcs = glob(["pre_compiled_models/*.tflite"]),
)

# ON DEVICE INTEGRATION TESTS ######################################################################

# NOTE: Everything here should be built with -c opt --config=android_arm64.

sh_binary(
    name = "run_on_device_driver_OSS",
    srcs = ["run_on_device_driver_OSS.sh"],
)

run_on_device(
    name = "single_op_device_tests_cpu",
    data = [":single_op_models_files"],
    driver = get_driver(),
    exec_args = [
        "--model_path={}".format(device_rlocation("single_op_models")),
        "--hw=cpu",
    ],
    exec_env_vars = [
        # TODO: Programatically set these and install proper vendor libraries.
        "LD_LIBRARY_PATH=/data/local/tmp",
        "ADSP_LIBRARY_PATH=/data/local/tmp",
    ],
    target = ":gen_device_test",
)

run_on_device(
    name = "single_op_device_tests_npu_JIT",
    data = [":single_op_models_files"],
    driver = get_driver(),
    exec_args = [
        "--model_path={}".format(device_rlocation("single_op_models")),
        "--hw=npu",
    ],
    exec_env_vars = [
        "LD_LIBRARY_PATH=/data/local/tmp",
        "ADSP_LIBRARY_PATH=/data/local/tmp",
    ],
    target = ":gen_device_test",
)

run_on_device(
    name = "classic_ml_device_tests_cpu",
    data = [":classic_ml_models"],
    driver = get_driver(),
    exec_args = [
        "--model_path={}".format(device_rlocation("classic_ml_models")),
        "--hw=cpu",
    ],
    exec_env_vars = [
        "LD_LIBRARY_PATH=/data/local/tmp",
        "ADSP_LIBRARY_PATH=/data/local/tmp",
    ],
    target = ":gen_device_test",
)

run_on_device(
    name = "classic_ml_device_tests_npu_JIT",
    data = [":classic_ml_models"],
    driver = get_driver(),
    exec_args = [
        "--model_path={}".format(device_rlocation("classic_ml_models")),
        "--hw=npu",
    ],
    exec_env_vars = [
        "LD_LIBRARY_PATH=/data/local/tmp",
        "ADSP_LIBRARY_PATH=/data/local/tmp",
    ],
    target = ":gen_device_test",
)

run_on_device(
    name = "pre_compiled_device_tests_qnn",
    data = [":pre_compiled_models"],
    driver = get_driver(),
    exec_args = [
        "--model_path={}".format(device_rlocation("pre_compiled_models")),
        "--hw=npu",
    ],
    exec_env_vars = [
        "LD_LIBRARY_PATH=/data/local/tmp",
        "ADSP_LIBRARY_PATH=/data/local/tmp",
    ],
    target = ":gen_device_test",
)
