# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

# We need special handling for JNI inclusion for Android. Rather than duplicating this logic
# for every target that uses JNI, we use a single proxy target that
# encapsulates it.
alias(
    name = "jni",
    actual = select({
        # The Android toolchain makes <jni.h> available in the system include
        # path.
        # Aliases need to resolve to a single target however, so alias to an
        # empty library instead.
        # (Making this target a cc_library with empty deps for the Android case
        # doesn't work, because go/cpp-features#layering-check requires targets
        # to _directly_ depend on libraries they include, and cc_library doesn't
        # have any direct equivalent to java_library's 'export' attribute).
        "//tensorflow:android": ":empty",
        # For non-Android toolchains, depend on the JDK JNI headers.
        "//conditions:default": "@bazel_tools//tools/jdk:jni",
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "empty",
    compatible_with = get_compatible_with_portable(),
)
