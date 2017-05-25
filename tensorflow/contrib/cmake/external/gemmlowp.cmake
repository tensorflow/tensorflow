# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
include (ExternalProject)

set(gemmlowp_URL http://github.com/google/gemmlowp/archive/a6f29d8ac48d63293f845f2253eccbf86bc28321.tar.gz)
set(gemmlowp_HASH SHA256=75d40ea8e68b0d1644f052fffe8f14a410b2a73d40ccb859a95c0578d194ec26)
set(gemmlowp_BUILD ${CMAKE_CURRENT_BINARY_DIR}/gemmlowp/src/gemmlowp)
set(gemmlowp_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gemmlowp/src/gemmlowp)

ExternalProject_Add(gemmlowp
    PREFIX gemmlowp
    URL ${gemmlowp_URL}
    URL_HASH ${gemmlowp_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/gemmlowp/CMakeLists.txt ${gemmlowp_BUILD}
    INSTALL_COMMAND "")
