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
    default_visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
)

# TODO: Put common copts/linkopts behind custom macro/rule.
QNN_SDK_LINKOPTS = [
    "-Wl,--dynamic-linker=/lib64/ld-linux-x86-64.so.2",
    "-Wl,--version-script=$(location {})".format(":lrt.lds"),
]

# NOTE: Currently the user needs to supply shared libraries from their own
# SDK.

# copybara:uncomment QNN_SDK_LIB_HTP_LINUX = "//third_party/qairt:lib/x86_64-linux-clang/libQnnHtp.so"

cc_library(
    name = "qnn_sdk_libcpp",
    srcs = [
        # copybara:uncomment "//third_party/qairt:lib/x86_64-linux-clang/libc++.so.1",
        # copybara:uncomment "//third_party/qairt:lib/x86_64-linux-clang/libc++abi.so.1",
    ],
    # TODO: Replace no builder with constraint on only linux systems.
    tags = ["nobuilder"],
)

cc_library(
    name = "load_sdk",
    srcs = ["load_sdk.cc"],
    hdrs = ["load_sdk.h"],
    # copybara:uncomment data = [QNN_SDK_LIB_HTP_LINUX],
    # copybara:uncomment defines = ["QNN_SDK_LIB_HTP=\\\"$(location {})\\\"".format(QNN_SDK_LIB_HTP_LINUX)],
    linkopts = [
        "-ldl",
    ],
    tags = ["nobuilder"],
    deps = [
        ":qnn_sdk_libcpp",  # buildcleaner: keep
        "@com_google_absl//absl/strings:string_view",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_manager",
    srcs = ["qnn_manager.cc"],
    hdrs = ["qnn_manager.h"],
    tags = ["nobuilder"],
    deps = [
        ":load_sdk",
        ":log",
        ":qnn_sdk_libcpp",  # buildcleaner: keep
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/core:api_internal",
    ],
)

cc_library(
    name = "log",
    srcs = ["log.cc"],
    hdrs = ["log.h"],
    tags = ["nobuilder"],
    deps = [
        ":qnn_sdk_libcpp",  # buildcleaner: keep
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
    ],
)

cc_library(
    name = "qnn_compiler_plugin",
    srcs = ["qnn_compiler_plugin.cc"],
    hdrs = ["//tensorflow/lite/experimental/lrt/c:lite_rt_compiler_plugin.h"],
    linkopts = QNN_SDK_LINKOPTS,
    tags = ["nobuilder"],
    deps = [
        ":lrt.lds",
        ":qnn_manager",
        ":qnn_sdk_libcpp",  # buildcleaner: keep
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
    ],
)

# NOTE: cc_test realistically will never link properly with the extra qnn libc++ needed. Need to find
# a long term solution for testing and packaging everything.
# TODO: In the short term we can wrap this in an "sh_test".
cc_binary(
    name = "qnn_compiler_pugin_manual_test",
    testonly = 1,
    srcs = [
        "qnn_compiler_plugin_manual_test.cc",
    ],
    linkopts = QNN_SDK_LINKOPTS,
    tags = ["nobuilder"],
    deps = [
        ":lrt.lds",
        ":qnn_compiler_plugin",  # buildcleaner: keep
        ":qnn_sdk_libcpp",  # buildcleaner: keep
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/test_data:test_data_util",
        "@com_google_absl//absl/log:absl_check",
    ],
)
