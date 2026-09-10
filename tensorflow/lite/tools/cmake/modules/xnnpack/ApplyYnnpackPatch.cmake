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

set(_YNNPACK_PATCH
  "${CMAKE_CURRENT_LIST_DIR}/../../../../../../third_party/xla/third_party/xnnpack/ynn_reduce_broadcast.patch"
)
get_filename_component(XNNPACK_SOURCE_DIR "${XNNPACK_SOURCE_DIR}" REALPATH)
get_filename_component(_XNNPACK_PARENT_DIR "${XNNPACK_SOURCE_DIR}" DIRECTORY)
# An archive may be extracted inside another Git checkout. Stop discovery at
# its parent so git apply does not silently skip paths outside that checkout's
# current subdirectory. A Git checkout of XNNPACK itself is still supported.
set(_YNNPACK_GIT
  "${CMAKE_COMMAND}" -E env "GIT_CEILING_DIRECTORIES=${_XNNPACK_PARENT_DIR}"
  "${GIT_EXECUTABLE}"
)

# Toggling YNNPACK off and on can rerun FetchContent's patch step. Accept only
# a fully applied patch; partial or incompatible contents must still fail.
execute_process(
  COMMAND ${_YNNPACK_GIT} apply --reverse --check "${_YNNPACK_PATCH}"
  WORKING_DIRECTORY "${XNNPACK_SOURCE_DIR}"
  RESULT_VARIABLE _YNNPACK_PATCH_APPLIED
  OUTPUT_QUIET ERROR_QUIET
)
if(_YNNPACK_PATCH_APPLIED EQUAL 0)
  return()
endif()

execute_process(
  COMMAND ${_YNNPACK_GIT} apply "${_YNNPACK_PATCH}"
  WORKING_DIRECTORY "${XNNPACK_SOURCE_DIR}"
  RESULT_VARIABLE _YNNPACK_PATCH_RESULT
  ERROR_VARIABLE _YNNPACK_PATCH_ERROR
)
if(NOT _YNNPACK_PATCH_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to apply YNNPACK reduction patch: ${_YNNPACK_PATCH_ERROR}")
endif()
