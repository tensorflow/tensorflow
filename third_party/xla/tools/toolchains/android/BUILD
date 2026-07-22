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

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

platform(
    name = "x86",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_32",
    ],
)

platform(
    name = "x86_64",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:x86_64",
    ],
)

platform(
    name = "armeabi-v7a",
    constraint_values = [
        "@platforms//os:android",
        "@platforms//cpu:armv7",
    ],
)

platform(
    name = "arm64-v8a",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:android",
    ],
)
