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

# 2D Fast Fourier Transform package
# from http://momonga.t.u-tokyo.ac.jp/~ooura/fft2d.html

package(
    default_visibility = ["//visibility:public"],
)

# Unrestricted use; can only distribute original package.
licenses(["notice"])

exports_files(["readme2d.txt"])

FFT2D_SRCS = [
    "fftsg.c",
    "fftsg2d.c",
]

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

# This is the main 2D FFT library.  The 2D FFTs in this library call
# 1D FFTs.  In addition, fast DCTs are provided for the special case
# of 8x8 and 16x16.  This code in this library is referred to as
# "Version II" on http://momonga.t.u-tokyo.ac.jp/~ooura/fft2d.html.
cc_library(
    name = "fft2d",
    srcs = FFT2D_SRCS,
    linkopts = select({
        ":windows": [],
        "//conditions:default": ["-lm"],
    }),
)

objc_library(
    name = "fft2d_ios",
    srcs = FFT2D_SRCS,
)

# Export the source code so that it could be compiled for Andoid native apps.
filegroup(
    name = "fft2d_srcs",
    srcs = FFT2D_SRCS,
)
