#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The "-mcpu=" switch might be introduced externaly into CMake: either in <LANG>_FLAGS or
# as part of CC, CXX, ASM environmental variables (to be stored in CMAKE_<LANG>_COMPILER_ARG1).
# This switch is not compatible with XNNPACK build mechanism and causes the XNNPACK compilation
# break due to "unsupported instructions". This switch needs to be removed for XNNPACK
# In order to isolate the changes only for XNNPACK and its depencencies, a subfolder is
# introduced.

foreach(FLAG IN ITEMS CMAKE_ASM_FLAGS CMAKE_ASM_COMPILER_ARG1 CMAKE_C_FLAGS CMAKE_C_COMPILER_ARG1 CMAKE_CXX_FLAGS CMAKE_CXX_COMPILER_ARG1)
  if(${FLAG})
    string(REGEX REPLACE "-mcpu=[-a-zA-Z0-9_.^$*+?]*" "" _tmp ${${FLAG}})
    set(${FLAG} ${_tmp})
  endif()
endforeach()

add_subdirectory(
  "${xnnpack_SOURCE_DIR}"
  "${xnnpack_BINARY_DIR}"
)
