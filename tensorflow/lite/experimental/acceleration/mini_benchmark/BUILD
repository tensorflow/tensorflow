# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# Mini benchmarking helps in validating HW acceleration across Android and iOS.
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow:tensorflow.bzl", "clean_dep")
load("build_defs.bzl", "embedded_binary")
load("special_rules.bzl", "jpeg_copts", "libjpeg_deps", "minibenchmark_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:__subpackages__",
        "//tensorflow/lite/tools/benchmark:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc:__subpackages__",
    ] + minibenchmark_visibility_allowlist(),
    licenses = ["notice"],
)

cc_library(
    name = "call",
    srcs = [
        "call.cc",
    ],
    hdrs = ["call_register.h"],
    copts = tflite_copts(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:op_macros",
        "@flatbuffers",
    ],
)

cc_test(
    name = "call_test",
    size = "small",
    srcs = ["call_test.cc"],
    deps = [
        ":call",
        "//tensorflow/lite:builtin_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:interpreter_test_util",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/kernels:subgraph_test_util",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

py_binary(
    name = "copy_associated_files",
    srcs = ["copy_associated_files.py"],
    python_version = "PY3",
    srcs_version = "PY3",
)

cc_library(
    name = "fb_storage",
    srcs = ["fb_storage.cc"],
    hdrs = ["fb_storage.h"],
    deps = [
        ":status_codes",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_test(
    name = "fb_storage_test",
    srcs = ["fb_storage_test.cc"],
    tags = ["no_windows"],  # Filesystem code not ported to windows.
    deps = [
        ":fb_storage",
        ":status_codes",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "jpeg_decompress_buffered_struct",
    hdrs = ["jpeg_decompress_buffered_struct.h"],
    copts = tflite_copts(),
    deps = [":libjpeg_hdrs"],
)

cc_test(
    name = "jpeg_decompress_buffered_struct_test",
    srcs = ["jpeg_decompress_buffered_struct_test.cc"],
    deps = [
        ":jpeg_decompress_buffered_struct",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "decode_jpeg_status",
    hdrs = [
        "decode_jpeg_status.h",
    ],
    deps = [
        "//tensorflow/lite/core/c:c_api_types",
    ],
)

cc_library(
    name = "libjpeg_handle",
    srcs = ["libjpeg_handle.cc"],
    hdrs = [
        "libjpeg_handle.h",
    ],
    copts = jpeg_copts(),
    deps = [
        ":decode_jpeg_status",
        ":jpeg_common",
        ":libjpeg_hdrs",
        "//tensorflow/lite/core/c:c_api_types",
    ],
)

cc_test(
    name = "libjpeg_handle_test",
    srcs = ["libjpeg_handle_test.cc"],
    linkopts = [
        "-ldl",
    ],
    deps = [
        ":decode_jpeg_status",
        ":libjpeg_handle",
        "//tensorflow/lite/core/c:c_api_types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "libc_handle",
    srcs = ["libc_handle.cc"],
    hdrs = ["libc_handle.h"],
    linkopts = select({
        # We use dlopen on Android.
        clean_dep("//tensorflow:android"): [
            "-ldl",
        ],
        "//conditions:default": [],
    }),
    deps = [":decode_jpeg_status"],
)

cc_test(
    name = "libc_handle_test",
    srcs = ["libc_handle_test.cc"],
    linkopts = [
        "-ldl",
    ],
    tags = [
        "no_mac",  # fmemopen not available on OS-X < 10.13.
        "no_windows",  # fmemopen not available on windows.
        "tflite_not_portable_ios",  # fmemopen not available on iOS < 11.
    ],
    deps = [
        ":decode_jpeg_status",
        ":libc_handle",
        "//tensorflow/lite/core/c:c_api_types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "libjpeg_hdrs",
    hdrs = ["libjpeg.h"],
    deps = libjpeg_deps(),
)

cc_library(
    name = "jpeg_common",
    hdrs = ["jpeg_common.h"],
    deps = [":libjpeg_hdrs"],
)

cc_library(
    name = "libjpeg_decoder",
    srcs = [
        "libjpeg_decoder.cc",
    ],
    hdrs = [
        "libjpeg_decoder.h",
    ],
    copts = tflite_copts(),
    deps = [
        ":decode_jpeg_status",
        ":jpeg_common",
        ":jpeg_decompress_buffered_struct",
        ":jpeg_header_parser",
        ":libc_handle",
        ":libjpeg_handle",
        ":libjpeg_hdrs",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "libjpeg_decoder_test_helper",
    testonly = 1,
    hdrs = ["libjpeg_decoder_test_helper.h"],
    linkopts = [
        "-lm",  # We included <cmath> in the source.
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "libjpeg_decoder_test",
    srcs = [
        "libjpeg_decoder_test.cc",
    ],
    copts = tflite_copts(),
    tags = [
        "no_mac",  # fmemopen not available on OS-X < 10.13.
        "no_windows",  # fmemopen not available on windows.
        "tflite_not_portable_ios",  # fmemopen not available on iOS < 11.
    ],
    deps = [
        ":decode_jpeg_status",
        ":embedded_chessboard_jpeg",
        ":embedded_snow_jpeg",
        ":embedded_test_card_jpeg",
        ":jpeg_header_parser",
        ":libjpeg_decoder",
        ":libjpeg_decoder_test_helper",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "decode_jpeg",
    srcs = [
        "decode_jpeg.cc",
    ],
    hdrs = ["decode_jpeg_register.h"],
    copts = tflite_copts(),
    deps = [
        ":libjpeg_decoder",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "@flatbuffers",
    ],
)

embedded_binary(
    name = "embedded_chessboard_jpeg",
    array_variable_name = "g_tflite_acceleration_chessboard_jpeg",
    binary = ":chessboard.jpg",
)

embedded_binary(
    name = "embedded_test_card_jpeg",
    array_variable_name = "g_tflite_acceleration_test_card_jpeg",
    binary = ":test_card.jpg",
)

embedded_binary(
    name = "embedded_snow_jpeg",
    array_variable_name = "g_tflite_acceleration_snow_jpeg",
    binary = "snow_4032_3024_3.jpg",
)

cc_test(
    name = "decode_jpeg_test",
    srcs = ["decode_jpeg_test.cc"],
    tags = [
        "no_mac",  # fmemopen not available on OS-X < 10.13.
        "no_windows",  # fmemopen not available on windows.
        "tflite_not_portable_ios",  # fmemopen not available on iOS < 11.
    ],
    deps = [
        ":decode_jpeg",
        ":embedded_chessboard_jpeg",
        ":embedded_test_card_jpeg",
        ":libjpeg_decoder_test_helper",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "jpeg_header_parser",
    srcs = ["jpeg_header_parser.cc"],
    hdrs = ["jpeg_header_parser.h"],
    deps = [
        ":decode_jpeg_status",
        ":jpeg_common",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:string",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "jpeg_header_parser_test",
    srcs = ["jpeg_header_parser_test.cc"],
    deps = [
        ":embedded_chessboard_jpeg",
        ":jpeg_header_parser",
        "//tensorflow/lite/core/c:c_api_types",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "odt_localizer_testfiles",
    srcs = [
        "data/flowers_192_192_3.jpeg",
        "data/person_192_192_3.jpeg",
        "data/rural_192_192_3.jpeg",
        "data/street_192_192_3.jpeg",
        "data/toys_192_192_3.jpeg",
    ],
)

filegroup(
    name = "odt_classifier_testfiles",
    srcs = [
        "data/chair_224_224_3.jpeg",
        "data/city_224_224_3.jpeg",
        "data/food_224_224_3.jpeg",
        "data/plant_224_224_3.jpeg",
        "data/shoe_224_224_3.jpeg",
    ],
)

filegroup(
    name = "facessd_testfiles",
    srcs = [
        "data/early_color_photo_faces_320_320_3.jpeg",
        "data/faces_in_painting_320_320_3.jpeg",
        "data/faces_in_snow_320_320_3.jpeg",
        "data/single_face_320_320_3.jpeg",
        "data/snowman_320_320_3.jpeg",
    ],
)

filegroup(
    name = "blazeface_testfiles",
    srcs = [
        "data/early_color_photo_face_128_224_3.jpeg",
        "data/face_in_painting_128_224_3.jpeg",
        "data/face_in_snow_128_224_3.jpeg",
        "data/single_face_128_224_3.jpeg",
        "data/snowman_128_224_3.jpeg",
    ],
)

cc_library(
    name = "status_codes",
    hdrs = ["status_codes.h"],
)

cc_library(
    name = "benchmark_result_evaluator",
    srcs = ["benchmark_result_evaluator.cc"],
    hdrs = ["benchmark_result_evaluator.h"],
    deps = ["//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs"],
)

cc_library(
    name = "validator",
    srcs = ["validator.cc"],
    hdrs = ["validator.h"],
    deps = [
        ":call",
        ":decode_jpeg",
        # Link in the GpuModulePlugin statically. It is created by name.
        ":gpu_module_plugin",  # buildcleaner: keep
        ":status_codes",
        "@com_google_absl//absl/container:flat_hash_set",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/c:c_api",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/core/experimental/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/core/kernels:builtin_ops",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:constants",
        "//tensorflow/lite/tools:model_loader",
    ],
)

cc_library(
    name = "validator_runner_options",
    srcs = ["validator_runner_options.cc"],
    hdrs = ["validator_runner_options.h"],
    deps = [
        ":benchmark_result_evaluator",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration/c:delegate_plugin",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",  # buildcleaner: keep
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
    ],
)

cc_library(
    name = "validator_runner_impl",
    srcs = ["validator_runner_impl.cc"],
    hdrs = ["validator_runner_impl.h"],
    deps = [
        ":benchmark_result_evaluator",
        ":fb_storage",
        ":file_lock",
        ":runner",
        ":status_codes",
        ":validator",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//tensorflow/lite/experimental/acceleration/configuration/c:delegate_plugin",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier:custom_validation_embedder",
        # For NNAPI support library, the headers and source files are defined
        # as two separate targets. We need to include both targets for NNAPI to
        # be invoked.
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",  # buildcleaner: keep
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "//tensorflow/lite/tools:model_loader",
    ],
)

cc_library(
    name = "blocking_validator_runner",
    srcs = ["blocking_validator_runner.cc"],
    hdrs = ["blocking_validator_runner.h"],
    deps = [
        ":status_codes",
        ":validator",
        ":validator_runner_impl",
        ":validator_runner_options",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@flatbuffers",
    ],
)

cc_library(
    name = "validator_runner",
    srcs = ["validator_runner.cc"],
    hdrs = ["validator_runner.h"],
    deps = [
        ":fb_storage",
        ":status_codes",
        ":validator",
        ":validator_runner_impl",
        ":validator_runner_options",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        # For NNAPI support library, the headers and source files are defined
        # as two separate targets. We need to include both targets for NNAPI to
        # be invoked.
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",  # buildcleaner: keep
    ],
)

# Version of validator_runner_entrypoint without definition of the
# SetBigCoresAffinity function.
# This allow us to override the SetBigCoresAffinity definition during test.
# We cannot simply alter the SetBigCoresAffinity behaviour to make it
# controllable by tests because the validation code will run in a separate
# process.
cc_library(
    name = "validator_runner_entrypoint_without_affinity_deps",
    srcs = ["validator_runner_entrypoint.cc"],
    hdrs = ["validator_runner_entrypoint.h"],
    deps = [
        ":constants",
        ":fb_storage",
        ":file_lock",
        ":set_big_core_affinity_h",
        ":status_codes",
        ":validator",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        # For NNAPI support library, the headears and source files are defined
        # as two separate targets. We need to include both targets for NNAPI to
        # be invoked.
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",  # buildcleaner: keep
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "//tensorflow/lite/tools:model_loader",
    ],
)

cc_library(
    name = "validator_runner_entrypoint",
    deps = [
        ":set_big_core_affinity",
        ":validator_runner_entrypoint_without_affinity_deps",
    ],
    alwayslink = 1,
)

embedded_binary(
    # Validation runner entry point binary that is embedded as constant data in code.
    # The embedded binary is extracted to disk before running.
    name = "embedded_validator_runner_entrypoint",
    testonly = True,
    array_variable_name = "g_tflite_acceleration_embedded_validator_runner_entrypoint",
    binary = ":validator_runner_so_for_tests",
)

cc_library(
    name = "set_big_core_affinity_h",
    hdrs = ["set_big_core_affinity.h"],
    deps = [":big_little_affinity"],
)

cc_library(
    name = "set_big_core_affinity",
    srcs = ["set_big_core_affinity.cc"],
    hdrs = ["set_big_core_affinity.h"],
    deps = [":big_little_affinity"],
    alwayslink = True,
)

#
# Infrastructure for running the mini-benchmark in its own process. See
# comments at the top of runner.cc for design notes.
#
# Production targets:
#
cc_binary(
    name = "runner_main_binary",
    srcs = ["runner_main.c"],
    copts = [
        # The following have no impact on binary size.
        # "-fno-exceptions",
        # "-ffunction-sections"
        # "-fdata-sections"
    ],
    linkopts = [
        "-ldl",
    ] + select({
        clean_dep("//tensorflow:debug"): [],
        "//conditions:default": [
            # Minimize binary size. Stripping cuts 64-bit binary
            # from 8kB to 6kB and 32-bit from 60k(!) to 6k.
            "-Wl,--strip-all",
            # The following have no real impact (16 bytes).
            #"-Wl,--no-export-dynamic",
            #"-Wl,--gc-sections",
            #"-Wl,--as-needed",
        ],
    }),
)

embedded_binary(
    # Validation runner binary that is embedded as constant data in code. The
    # embedded binary is extracted to disk before running.
    name = "embedded_runner_executable",
    array_variable_name = "g_tflite_acceleration_embedded_runner",
    binary = ":runner_main_binary",
)

cc_library(
    name = "runner",
    srcs = ["runner.cc"],
    hdrs = ["runner.h"],
    deps = [
        ":big_little_affinity",
        ":constants",
        ":status_codes",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:stderr_reporter",
    ] + select({
        clean_dep("//tensorflow:android"): [
            ":embedded_runner_executable",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "big_little_affinity",
    srcs = ["big_little_affinity.cc"],
    hdrs = ["big_little_affinity.h"],
    deps = [
        "@cpuinfo//:cpuinfo_with_unstripped_include_path",
    ],
)

cc_test(
    name = "big_little_affinity_test",
    srcs = ["big_little_affinity_test.cc"],
    tags = ["no_windows"],  # Not ported to windows.
    deps = [
        ":big_little_affinity",
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "@com_google_googletest//:gtest_main",
        "@cpuinfo//:cpuinfo_with_unstripped_include_path",
    ],
)

cc_library(
    name = "mini_benchmark",
    srcs = [
        "mini_benchmark.cc",
    ],
    hdrs = ["mini_benchmark.h"],
    visibility = [
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:__subpackages__",
        "//tensorflow/lite/tools/benchmark:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc:__subpackages__",
    ] + minibenchmark_visibility_allowlist(),
    deps = [
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@flatbuffers",
    ],
)

cc_library(
    name = "mini_benchmark_implementation",
    srcs = [
        "mini_benchmark_implementation.cc",
    ],
    visibility = [
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:__subpackages__",
        "//tensorflow/lite/tools/benchmark:__subpackages__",
        "@org_tensorflow_lite_support//tensorflow_lite_support/cc:__subpackages__",
    ] + minibenchmark_visibility_allowlist(),
    deps = [
        ":fb_storage",
        ":mini_benchmark",
        ":status_codes",
        ":validator_runner",
        ":validator_runner_entrypoint",
        ":validator_runner_options",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

embedded_binary(
    name = "embedded_mobilenet_float_validation_model",
    testonly = 1,
    array_variable_name = "g_tflite_acceleration_embedded_mobilenet_float_validation_model",
    binary = "//tensorflow/lite/experimental/acceleration/mini_benchmark/metrics:mobilenet_float_with_validation.tflite",
)

cc_library(
    name = "mini_benchmark_test_helper",
    testonly = 1,
    srcs = ["mini_benchmark_test_helper.cc"],
    hdrs = ["mini_benchmark_test_helper.h"],
    linkopts = select({
        # We will use dlopen on Android.
        clean_dep("//tensorflow:android"): ["-ldl"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest",
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "//tensorflow/lite/tools:logging",
    ] + select({
        clean_dep("//tensorflow:android"): [
            ":embedded_runner_executable",
            ":embedded_validator_runner_entrypoint",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "mini_benchmark_test",
    srcs = [
        "mini_benchmark_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":embedded_mobilenet_float_validation_model",
        ":mini_benchmark",
        ":mini_benchmark_implementation",
        ":mini_benchmark_test_helper",
        ":status_codes",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_cc_proto",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/experimental/acceleration/configuration:proto_to_flatbuffer",
        "//tensorflow/lite/experimental/acceleration/configuration:xnnpack_plugin",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:embedded_nnapi_sl_fake_impl",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark:nnapi_sl_fake_impl_client",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "file_lock",
    srcs = ["file_lock.cc"],
    hdrs = ["file_lock.h"],
)

cc_test(
    name = "file_lock_test",
    srcs = ["file_lock_test.cc"],
    deps = [
        ":file_lock",
        "@com_google_googletest//:gtest_main",
    ],
)

#
# Test targets for separate process.
# Unit tests using cc_test and turned into Android tests with tflite_portable_test_suite().

cc_binary(
    name = "runner_unit_test_entry_points",
    srcs = ["runner_test_entry_points.cc"],
    linkshared = True,
    deps = [
        ":constants",
        ":status_codes",
        "//tensorflow/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/lite/tools:model_loader",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_library(
    name = "runner_unit_test_entry_points_library",
    srcs = ["runner_test_entry_points.cc"],
    deps = [
        ":constants",
        ":status_codes",
        "//tensorflow/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/lite/tools:model_loader",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
    ],
)

embedded_binary(
    # Embed the runner_unit_test_entry_points binary for extracting in the unit
    # test. This is needed so that the unit test and the binary are built for
    # the same architecture (as opposed to having as a data dependency, which
    # get built only for the first architecture with fat apk).
    name = "embedded_runner_unit_test_entry_points",
    array_variable_name = "g_tflite_acceleration_embedded_runner_unit_test_entry_points",
    binary = "runner_unit_test_entry_points",
)

cc_test(
    name = "runner_test",
    srcs = ["runner_test.cc"],
    linkopts = [
        "-ldl",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":runner",
        ":status_codes",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "//tensorflow/lite/schema:schema_fbs_with_mutable",
    ] + select({
        clean_dep("//tensorflow:android"): [
            ":embedded_runner_executable",
            ":embedded_runner_unit_test_entry_points",
            ":mini_benchmark_test_helper",
        ],
        "//conditions:default": [
            ":runner_unit_test_entry_points_library",
        ],
    }),
)

cc_binary(
    name = "validator_runner_so_for_tests",
    testonly = 1,
    srcs = [
        "validator_runner_entrypoint.cc",
        "validator_runner_entrypoint.h",
    ],
    linkshared = True,
    deps = [
        ":constants",
        ":fb_storage",
        ":runner",
        ":status_codes",
        ":set_big_core_affinity",
        ":file_lock",
        ":validator",
        ":validator_runner",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "//tensorflow/lite/tools:model_loader",
        "//tensorflow/lite:minimal_logging",
    ] + select({
        # On Android, as the validation runs in a separate process as a
        # different binary, any TFLite delegates to be validated need to
        # include corresponding delegate plugins.
        clean_dep("//tensorflow:android"): [
            "//tensorflow/lite/experimental/acceleration/configuration:gpu_plugin",
            "//tensorflow/lite/experimental/acceleration/configuration:xnnpack_plugin",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "nnapi_sl_fake_impl_client",
    testonly = 1,
    srcs = ["nnapi_sl_fake_impl.cc"],
    hdrs = ["nnapi_sl_fake_impl.h"],
    deps = [
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
    ],
)

cc_binary(
    name = "nnapi_sl_fake_impl",
    testonly = 1,
    srcs = ["nnapi_sl_fake_impl.cc"],
    linkshared = True,
    deps = [
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
    ],
)

cc_library(
    name = "gpu_module_plugin",
    srcs = ["gpu_module_plugin.cc"],
    hdrs = ["gpu_module_plugin.h"],
    deps = [
        ":status_codes",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/core/experimental/acceleration/configuration:delegate_registry",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration/c:delegate_plugin",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

embedded_binary(
    name = "embedded_nnapi_sl_fake_impl",
    testonly = 1,
    array_variable_name = "g_nnapi_sl_fake_impl",
    binary = ":nnapi_sl_fake_impl",
)

embedded_binary(
    name = "embedded_mobilenet_validation_model",
    testonly = 1,
    array_variable_name = "g_tflite_acceleration_embedded_mobilenet_validation_model",
    binary = "//tensorflow/lite/experimental/acceleration/mini_benchmark/metrics:mobilenet_quant_with_validation.tflite",
)

embedded_binary(
    name = "embedded_mobilenet_model",
    testonly = 1,
    array_variable_name = "g_tflite_acceleration_embedded_mobilenet_model",
    binary = "//tensorflow/lite/experimental/acceleration/mini_benchmark/models:mobilenet_v1_1.0_224_quant.tflite",
)

cc_test(
    name = "validator_test",
    srcs = ["validator_test.cc"],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":embedded_mobilenet_validation_model",
        ":embedded_mobilenet_model",
        ":mini_benchmark_test_helper",
        ":status_codes",
        ":validator",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
        "@flatbuffers",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier:custom_validation_embedder",
        "//tensorflow/lite/schema:schema_fbs_with_mutable",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_cc_proto",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration:flatbuffer_to_proto",
        "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/experimental/acceleration/configuration:proto_to_flatbuffer",
        "//tensorflow/lite/tools:model_loader",
        "//tensorflow/lite:stderr_reporter",
    ] + select({
        clean_dep("//tensorflow:android"): [
            "//tensorflow/lite/experimental/acceleration/configuration:gpu_plugin",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "validator_runner_impl_test",
    srcs = ["validator_runner_impl_test.cc"],
    deps = [
        ":benchmark_result_evaluator",
        ":embedded_mobilenet_model",
        ":embedded_mobilenet_validation_model",
        ":embedded_nnapi_sl_fake_impl",
        ":embedded_validator_runner_entrypoint",
        ":fb_storage",
        ":mini_benchmark_test_helper",
        ":nnapi_sl_fake_impl_client",
        ":status_codes",
        ":validator_runner_entrypoint",
        ":validator_runner_impl",
        ":validator_runner_options",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier:custom_validation_embedder",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",  # buildcleaner: keep
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "blocking_validator_runner_test",
    srcs = ["blocking_validator_runner_test.cc"],
    deps = [
        ":benchmark_result_evaluator",
        ":blocking_validator_runner",
        ":embedded_mobilenet_model",
        ":embedded_mobilenet_validation_model",
        ":mini_benchmark_test_helper",
        ":status_codes",
        ":validator_runner_entrypoint",
        ":validator_runner_options",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_test(
    name = "validator_runner_test",
    srcs = ["validator_runner_test.cc"],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":embedded_mobilenet_validation_model",
        ":embedded_nnapi_sl_fake_impl",
        ":mini_benchmark_test_helper",
        ":nnapi_sl_fake_impl_client",
        ":status_codes",
        ":validator_runner",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "//tensorflow/lite/experimental/acceleration/compatibility:android_info",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library",
        "//tensorflow/lite/nnapi/sl:nnapi_support_library_headers",
    ] + select({
        clean_dep("//tensorflow:android"): [
            "//tensorflow/lite/experimental/acceleration/configuration:gpu_plugin",
        ],
        "//conditions:default": [
            ":validator_runner_entrypoint",
        ],
    }),
)

cc_test(
    name = "validator_runner_entrypoint_test",
    srcs = [
        "set_big_core_affinity.h",
        "validator_runner_entrypoint_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":fb_storage",
        ":status_codes",
        ":validator",
        # This target is used with dlsym() in the test.
        ":validator_runner_entrypoint_without_affinity_deps",  # buildcleaner: keep
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
    ],
)

exports_files(["model_validation_test.cc"])

tflite_portable_test_suite()
