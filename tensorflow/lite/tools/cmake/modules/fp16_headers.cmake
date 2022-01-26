#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Skip installation when xnnpack is used since it also has fp16 headers.
if(TARGET fp16_headers OR fp16_headers_POPULATED OR TFLITE_ENABLE_XNNPACK)
  return()
endif()

include(utils)
# Parsing of the fp16_headers archive URL and its checksum requires slightly 
# different logic compared to the rest of TensorFlow Lite CMake modules,
# rendering most of the 'utils' module helper functions unusable, therefore
# requiring the implementation below.
set(REMOTE_CONF_FILE "https://raw.githubusercontent.com/google/XNNPACK/master/cmake/DownloadFP16.cmake")
set(TEMP_CONF_PATH "${CMAKE_BINARY_DIR}/temp_DownloadFP16.cmake")
file(DOWNLOAD ${REMOTE_CONF_FILE} ${TEMP_CONF_PATH})

# Retrieve fp16_headers archive URL
file(STRINGS ${TEMP_CONF_PATH}
    URL_LINE
    REGEX "^[ ]*URL[ ]+http[a-zA-Z0-9/.:_-]*$"
    LIMIT_COUNT 1
)
if(NOT URL_LINE)
    print_parse_error("fp16_headers" ${REMOTE_CONF_FILE} "URL")
endif()

string(STRIP ${URL_LINE} URL_LINE)

# Retrieve dependency archive checksum
file(STRINGS ${TEMP_CONF_PATH}
    CHECKSUM_LINE
    REGEX "^[ ]*URL_HASH[ ]+SHA256[ ]*=[ ]*[a-zA-Z0-9]*$"
    LIMIT_COUNT 1
)

if(NOT CHECKSUM_LINE)
    print_parse_error("fp16_headers" ${REMOTE_CONF_FILE} "URL_HASH SHA256")
endif()

string(STRIP ${CHECKSUM_LINE} CHECKSUM_LINE)

# Extract dependency archive URL value
string(FIND ${URL_LINE} "http" URL_START)
string(SUBSTRING ${URL_LINE} ${URL_START} -1 FP16_HEADERS_URL)

# Extract SHA-256 value
string(FIND ${CHECKSUM_LINE} "=" EQUAL_POS)
math(EXPR SHA256_START "${EQUAL_POS}+1")
string(SUBSTRING ${CHECKSUM_LINE} ${SHA256_START} -1 FP16_HEADERS_CHECKSUM)
check_sha_length("fp16_headers" ${REMOTE_CONF_FILE} "URL_HASH SHA256" ${FP16_HEADERS_CHECKSUM} 32)

file(REMOVE ${TEMP_CONF_PATH})
message(STATUS "Cloning fp16_headers repository from ${FP16_HEADERS_URL}, found in ${REMOTE_CONF_FILE}...")

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  fp16_headers
  # Automatically synced with https://github.com/google/XNNPACK/blob/master/cmake/DownloadFP16.cmake
  URL ${FP16_HEADERS_URL}
  URL_HASH SHA256=${FP16_HEADERS_CHECKSUM}
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/fp16_headers"
)

OverridableFetchContent_GetProperties(fp16_headers)
if(NOT fp16_headers)
  OverridableFetchContent_Populate(fp16_headers)
endif()

include_directories(
  AFTER
   "${fp16_headers_SOURCE_DIR}/include"
)
