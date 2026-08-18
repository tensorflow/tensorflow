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

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "gcc",
    srcs = glob(["bin/*-gcc"]),
)

filegroup(
    name = "ar",
    srcs = glob(["bin/*-ar"]),
)

filegroup(
    name = "ld",
    srcs = glob(["bin/*-ld"]),
)

filegroup(
    name = "nm",
    srcs = glob(["bin/*-nm"]),
)

filegroup(
    name = "objcopy",
    srcs = glob(["bin/*-objcopy"]),
)

filegroup(
    name = "objdump",
    srcs = glob(["bin/*-objdump"]),
)

filegroup(
    name = "strip",
    srcs = glob(["bin/*-strip"]),
)

filegroup(
    name = "as",
    srcs = glob(["bin/*-as"]),
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "arm-rpi-linux-gnueabihf/**",
        "libexec/**",
        "lib/gcc/arm-rpi-linux-gnueabihf/**",
        "include/**",
    ]),
)

filegroup(
    name = "aarch64_compiler_pieces",
    srcs = glob([
        "aarch64-none-linux-gnu/**",
        "libexec/**",
        "lib/gcc/aarch64-none-linux-gnu/**",
        "include/**",
    ]),
)

filegroup(
    name = "compiler_components",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)
