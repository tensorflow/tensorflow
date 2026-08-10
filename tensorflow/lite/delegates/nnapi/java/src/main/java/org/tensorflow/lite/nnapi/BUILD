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

# WARNING: this package is DEPRECATED.
# See https://developer.android.com/ndk/guides/neuralnetworks/migration-guide.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

filegroup(
    name = "nnapi_delegate_src",
    srcs = [
        "NnApiDelegate.java",
        "NnApiDelegateImpl.java",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nnapi_delegate_api_src",
    srcs = ["NnApiDelegate.java"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "nnapi_delegate_impl_src",
    srcs = ["NnApiDelegateImpl.java"],
    visibility = ["//visibility:public"],
)
