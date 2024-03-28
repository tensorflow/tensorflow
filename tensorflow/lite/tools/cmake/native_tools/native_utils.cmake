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

# Checks for presence of natively-built compiler for cross-compilation purposes
macro(report_found CNAME BIN_PATH)
  message(STATUS "Natively pre-built '${CNAME}' compiler for cross-compilation purposes found: ${BIN_PATH}")
endmacro()

macro(report_missing CNAME SEARCHED_PATHS)
  message(FATAL_ERROR "Natively compiled '${CNAME}' compiler has not been found in the following\
  locations: ${SEARCHED_PATHS}")
endmacro()

function (find_native_compiler CNAME)
  set(CANDIDATE_CPATHS
      ${TFLITE_HOST_TOOLS_DIR}/bin
      ${TFLITE_HOST_TOOLS_DIR}
      )
  
  if(${CNAME} STREQUAL "flatc")
    find_program(FLATC-BIN ${CNAME} PATHS ${CANDIDATE_CPATHS} NO_DEFAULT_PATH)
    if(${FLATC-BIN} STREQUAL "FLATC-BIN-NOTFOUND")
      report_missing(${CNAME} ${CANDIDATE_CPATHS})
    else()
      report_found(${CNAME} ${FLATC-BIN})
    endif()
  endif()

  if(${CNAME} STREQUAL "protoc")
    find_program(PROTOC-BIN ${CNAME} PATHS ${CANDIDATE_CPATHS} NO_DEFAULT_PATH)
    if(${PROTOC-BIN} STREQUAL "PROTOC-BIN-NOTFOUND")
      report_missing(${CNAME} ${CANDIDATE_CPATHS})
    else()
      report_found(${CNAME} ${PROTOC-BIN})
    endif()
  endif()

endfunction()