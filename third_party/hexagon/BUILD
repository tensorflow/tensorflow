# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

package(default_visibility = [
    "//visibility:public",
])

licenses([
    "notice",  # BSD-3-Clause-Clear
])

exports_files(glob(["hexagon/**/*.so"]))

#Just header file, needed for data types in the interface.
cc_library(
    name = "hexagon_nn_header",
    hdrs = [
        "hexagon/hexagon_nn.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)

cc_library(
    name = "hexagon_nn_ops",
    hdrs = [
        "hexagon/hexagon_nn_ops.h",
        "hexagon/ops.def",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)
