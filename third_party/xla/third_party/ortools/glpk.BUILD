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

cc_library(
    name = "glpk",
    srcs = glob(
        [
            "glpk-4.52/src/*.c",
            "glpk-4.52/src/*/*.c",
            "glpk-4.52/src/*.h",
            "glpk-4.52/src/*/*.h",
        ],
        exclude = ["glpk-4.52/src/proxy/main.c"],
    ),
    hdrs = [
        "glpk-4.52/src/glpk.h",
    ],
    copts = [
        "-Wno-error",
        "-w",
        "-Iexternal/glpk/glpk-4.52/src",
        "-Iexternal/glpk/glpk-4.52/src/amd",
        "-Iexternal/glpk/glpk-4.52/src/bflib",
        "-Iexternal/glpk/glpk-4.52/src/cglib",
        "-Iexternal/glpk/glpk-4.52/src/colamd",
        "-Iexternal/glpk/glpk-4.52/src/env",
        "-Iexternal/glpk/glpk-4.52/src/minisat",
        "-Iexternal/glpk/glpk-4.52/src/misc",
        "-Iexternal/glpk/glpk-4.52/src/proxy",
        "-Iexternal/glpk/glpk-4.52/src/zlib",
        "-DHAVE_ZLIB",
    ],
    includes = ["glpk-4.52/src"],
    visibility = ["//visibility:public"],
)
