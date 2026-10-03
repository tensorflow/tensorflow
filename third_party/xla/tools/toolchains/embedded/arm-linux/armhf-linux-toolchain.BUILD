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
    srcs = [
        "bin/arm-none-linux-gnueabihf-gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "bin/arm-none-linux-gnueabihf-ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "bin/arm-none-linux-gnueabihf-ld",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "bin/arm-none-linux-gnueabihf-nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "bin/arm-none-linux-gnueabihf-objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "bin/arm-none-linux-gnueabihf-objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "bin/arm-none-linux-gnueabihf-strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "bin/arm-none-linux-gnueabihf-as",
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "arm-none-linux-gnueabihf/**",
        "libexec/**",
        "lib/gcc/arm-none-linux-gnueabihf/**",
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
