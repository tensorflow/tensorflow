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

# A sample delegate that supports addition and subtraction only.
load("//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object", "tflite_copts", "tflite_linkopts_no_undefined")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "sample_stable_delegate",
    testonly = True,
    srcs = [
        "sample_stable_delegate.cc",
    ],
    hdrs = [
        "sample_stable_delegate.h",
    ],
    copts = tflite_copts(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_test(
    name = "sample_stable_delegate_test",
    srcs = ["sample_stable_delegate_test.cc"],
    data = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/sub.bin",
    ],
    deps = [
        ":sample_stable_delegate_opaque_delegate",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "sample_stable_delegate_external",
    testonly = True,
    srcs = [
        "sample_stable_delegate_external.cc",
    ],
    copts = tflite_copts(),
    generate_opaque_delegate_target = True,
    tflite_deps = [
        ":sample_stable_delegate",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_opaque_delegate",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:stable_delegate_interface",
        "//tensorflow/lite/experimental/acceleration/configuration/c:delegate_plugin",
        "//tensorflow/lite/experimental/acceleration/configuration/c:stable_delegate",
    ],
)

tflite_cc_shared_object(
    name = "tensorflowlite_sample_stable_delegate",
    testonly = True,
    linkopts = tflite_linkopts_no_undefined() + select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            # Expose necessary symbols only.
            "-Wl,--version-script,$(location //tensorflow/lite/delegates/utils/experimental/stable_delegate:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":sample_stable_delegate_external_opaque_delegate",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:version_script.lds",
    ],
)

cc_test(
    name = "sample_stable_delegate_external_test",
    srcs = ["sample_stable_delegate_external_test.cc"],
    data = [
        ":tensorflowlite_sample_stable_delegate",
        "//tensorflow/lite:testdata/add.bin",
    ],
    deps = [
        ":sample_stable_delegate_opaque_delegate",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "sample_stable_delegate_kernel_tests",
    timeout = "moderate",
    args = ["--stable_delegate_settings_file=$(location stable_delegate_settings.json)"],
    data = [
        "stable_delegate_settings.json",
        ":tensorflowlite_sample_stable_delegate",
    ],
    # This shard_count value of 30 was chosen because empirically this seems to be near the sweet
    # spot that minimizes test time while not wasting resources with unnecessary shards.
    # This runs roughly 30% faster than with shard_count = 20, but values beyond this show little
    # improvement, e.g. shard_count = 40 gave only a few per cent improvement.
    shard_count = 30,
    deps = [
        "//tensorflow/lite/kernels:combined_all_kernel_tests_lib",
        "//tensorflow/lite/kernels:test_main",
    ],
)

genrule(
    name = "gen_stable_delegate_settings",
    outs = ["stable_delegate_settings.json"],
    cmd = """
        echo '
            {
              "stable_delegate_loader_settings": {
                "delegate_path": "tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/libtensorflowlite_sample_stable_delegate.so"
              }
            }
        ' > $@
    """,
)
