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
#   libpng is the official PNG reference library.

licenses(["notice"])  # BSD/MIT-like license

exports_files(["LICENSE"])

cc_library(
    name = "png",
    srcs = [
        "png.c",
        "pngdebug.h",
        "pngerror.c",
        "pngget.c",
        "pnginfo.h",
        "pnglibconf.h",
        "pngmem.c",
        "pngpread.c",
        "pngpriv.h",
        "pngread.c",
        "pngrio.c",
        "pngrtran.c",
        "pngrutil.c",
        "pngset.c",
        "pngstruct.h",
        "pngtrans.c",
        "pngwio.c",
        "pngwrite.c",
        "pngwtran.c",
        "pngwutil.c",
    ] + select({
        ":windows": [
            "intel/filter_sse2_intrinsics.c",
            "intel/intel_init.c",
        ],
        "@xla//xla/tsl:linux_ppc64le": [
            "powerpc/filter_vsx_intrinsics.c",
            "powerpc/powerpc_init.c",
        ],
        "//conditions:default": [
        ],
    }),
    hdrs = [
        "png.h",
        "pngconf.h",
    ],
    copts = select({
        ":windows": ["-DPNG_INTEL_SSE_OPT=1"],
        "//conditions:default": [],
    }),
    includes = ["."],
    linkopts = select({
        ":windows": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//visibility:public"],
    deps = ["@zlib"],
)

genrule(
    name = "snappy_stubs_public_h",
    srcs = ["scripts/pnglibconf.h.prebuilt"],
    outs = ["pnglibconf.h"],
    cmd = "sed -e 's/PNG_ZLIB_VERNUM 0/PNG_ZLIB_VERNUM 0x1310/' $< >$@",
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)
