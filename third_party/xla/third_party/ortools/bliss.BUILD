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
    name = "libbliss",
    srcs = [
        "bliss-0.73/bliss_C.cc",
        "bliss-0.73/defs.cc",
        "bliss-0.73/graph.cc",
        "bliss-0.73/heap.cc",
        "bliss-0.73/orbit.cc",
        "bliss-0.73/partition.cc",
        "bliss-0.73/timer.cc",
        "bliss-0.73/uintseqhash.cc",
        "bliss-0.73/utils.cc",
    ],
    hdrs = [
        "bliss-0.73/bignum.hh",
        "bliss-0.73/bliss_C.h",
        "bliss-0.73/defs.hh",
        "bliss-0.73/graph.hh",
        "bliss-0.73/heap.hh",
        "bliss-0.73/kqueue.hh",
        "bliss-0.73/kstack.hh",
        "bliss-0.73/orbit.hh",
        "bliss-0.73/partition.hh",
        "bliss-0.73/timer.hh",
        "bliss-0.73/uintseqhash.hh",
        "bliss-0.73/utils.hh",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
