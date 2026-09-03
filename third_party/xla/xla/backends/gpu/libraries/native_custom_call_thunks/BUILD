# Copyright 2026 The OpenXLA Authors.
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
#
# Thunk-folding API for GPU custom calls: lets custom calls lower directly to native ThunkSequences
# instead of being wrapped in a CustomCallThunk (FFI/legacy). See
# `native_custom_call_handler_registry.h`.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# Visibility allowlist for packages that are permitted to register custom-call
# thunk-folding handlers.
package_group(
    name = "handler_allowlist",
    packages = [
        # Demonstrator / test-only handler.
        "//xla/backends/gpu/libraries/native_custom_call_thunks/...",
    ],
)

cc_library(
    name = "native_custom_call_emitter_context",
    hdrs = ["native_custom_call_emitter_context.h"],
    visibility = [
        ":handler_allowlist",
        "//xla/service/gpu:__pkg__",
    ],
    deps = [
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/ffi:attribute_map",
        "//xla/service:buffer_assignment",
        "//xla/service:gpu_topology",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "native_custom_call_handler_registry",
    srcs = ["native_custom_call_handler_registry.cc"],
    hdrs = ["native_custom_call_handler_registry.h"],
    visibility = ["//xla:internal"],
    deps = [
        ":native_custom_call_emitter_context",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "native_custom_call_handler_registration",
    srcs = ["native_custom_call_handler_registration.cc"],
    hdrs = ["native_custom_call_handler_registration.h"],
    visibility = [":handler_allowlist"],
    deps = [
        ":native_custom_call_handler_registry",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "native_custom_call_handler_registry_test",
    srcs = ["native_custom_call_handler_registry_test.cc"],
    deps = [
        ":native_custom_call_emitter_context",
        ":native_custom_call_handler_registration",
        ":native_custom_call_handler_registry",
        "//xla:shape_util",
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:thunk_id",
        "//xla/ffi:attribute_map",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:gpu_topology",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)
