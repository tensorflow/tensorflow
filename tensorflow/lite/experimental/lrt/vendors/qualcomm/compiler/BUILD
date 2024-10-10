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

load("//tensorflow/lite/experimental/lrt/build_common:lite_rt_build_defs.bzl", "lite_rt_dynamic_lib", "lite_rt_lib", "lite_rt_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

lite_rt_dynamic_lib(
    name = "qnn_compiler_plugin",
    srcs = ["qnn_compiler_plugin.cc"],
    hdrs = ["//tensorflow/lite/experimental/lrt/c:lite_rt_compiler_plugin.h"],
    export_lrt_only = True,
    shared_lib_name = "qnn_compiler_plugin_so",
    so_name = "libLrtQnnCompilerPlugin.so",
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    ungrte = True,
    visibility = ["//tensorflow/lite/experimental/lrt:__subpackages__"],
    deps = [
        ":qnn_compose_graph",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/vendors/qualcomm:qnn_manager",
    ],
)

lite_rt_test(
    name = "qnn_compiler_plugin_test",
    srcs = [
        "qnn_compiler_plugin_test.cc",
    ],
    linkstatic = True,
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    use_sys_malloc = True,
    deps = [
        ":qnn_compiler_plugin",  # buildcleaner: keep
        "@com_google_absl//absl/log:absl_check",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/core:api_internal",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/core:model",
        "//tensorflow/lite/experimental/lrt/test:common",
        "//tensorflow/lite/experimental/lrt/vendors/qualcomm:qnn_manager",
    ],
)

lite_rt_lib(
    name = "qnn_compose_graph",
    srcs = ["qnn_compose_graph.cc"],
    hdrs = ["qnn_compose_graph.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment "//third_party/qairt:qnn_lib_headers",
        "//tensorflow/lite/experimental/lrt/c:lite_rt_c_api",
        "//tensorflow/lite/experimental/lrt/cc:lite_rt_cc_api",
        "//tensorflow/lite/experimental/lrt/core:graph_tools",
        "//tensorflow/lite/experimental/lrt/vendors/qualcomm:qnn_manager",
        "//tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/IR:qnn_op",
        "//tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/IR:qnn_tensor",
    ],
)
