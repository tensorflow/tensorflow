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

# Description:
#   A library for decoding and encoding GIF images

licenses(["notice"])  # MIT

exports_files(["COPYING"])

cc_library(
    name = "gif",
    srcs = [
        "dgif_lib.c",
        "egif_lib.c",
        "gif_err.c",
        "gif_font.c",
        "gif_hash.c",
        "gif_hash.h",
        "gif_lib_private.h",
        "gifalloc.c",
        "openbsd-reallocarray.c",
        "quantize.c",
    ],
    hdrs = ["gif_lib.h"],
    defines = select({
        ":android": [
            "S_IREAD=S_IRUSR",
            "S_IWRITE=S_IWUSR",
            "S_IEXEC=S_IXUSR",
        ],
        "//conditions:default": [],
    }),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = select({
        ":windows": [":windows_polyfill"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "windows_polyfill",
    hdrs = ["windows/unistd.h"],
    includes = ["windows"],
)

genrule(
    name = "windows_unistd_h",
    outs = ["windows/unistd.h"],
    cmd = "touch $@",
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
)
