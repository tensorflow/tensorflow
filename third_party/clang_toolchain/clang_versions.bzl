# Copyright 2024 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLVM Clang versions."""

LLVM_DOWNLOADS_PREFIX = "https://github.com/llvm/llvm-project/releases/download/llvmorg-{}"

DEFAULT_CLANG_VERSION = "18.1.8"

CLANG_VERSIONS_DICT = {
    "18.1.8": {
        "windows": ["clang+llvm-18.1.8-x86_64-pc-windows-msvc.tar.xz", ""],
        "mac os": ["clang+llvm-18.1.8-arm64-apple-macos11.tar.xz", ""],
        "linux_aarch64": ["clang+llvm-18.1.8-aarch64-linux-gnu.tar.xz", ""],
        "linux_amd64": ["clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz", ""],
    },
}
