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

macro(print_parse_error
    MODULE
    BAZEL_DEF_FILE
    BAZEL_VAR_NAME
)
    string(TOUPPER ${MODULE} MODULE_UPPERCASE)
    message(FATAL_ERROR "${MODULE} module: could not retrieve valid\
    ${BAZEL_VAR_NAME} variable value from the following Bazel configuration\
    file: ${BAZEL_DEF_FILE}.")
endmacro()

macro(check_sha_length
    MODULE
    BAZEL_DEF_FILE
    BAZEL_VAR_NAME
    SHA
    EXPECTED_LEN_BYTES
)
    math(EXPR EXPECTED_SHA_LEN "${EXPECTED_LEN_BYTES}*2")
    string(LENGTH ${SHA} SHA_LEN)
    if(NOT ${SHA_LEN} EQUAL ${EXPECTED_SHA_LEN})
        message(FATAL_ERROR "Invalid length (${SHA_LEN}) of ${BAZEL_VAR_NAME} value\
        located in ${BAZEL_DEF_FILE}. Expected length: ${EXPECTED_SHA_LEN} ")
    endif()
endmacro()

function(extract_quote_contents
    PARSED_LINE 
    QUOTE_CONTENTS
)
    string(FIND ${PARSED_LINE} "\"" SHA_START)
    string(SUBSTRING ${PARSED_LINE} ${SHA_START} -1 QUOTED_SHA)
    string(LENGTH ${QUOTED_SHA} QUOTED_SHA_LEN)
    math(EXPR QUOTED_SHA_LEN "${QUOTED_SHA_LEN}-2")
    string(SUBSTRING ${QUOTED_SHA} 1 ${QUOTED_SHA_LEN} RAW_SHA)

    set(${QUOTE_CONTENTS} ${RAW_SHA} PARENT_SCOPE)
endfunction()


# Parses the <BAZEL_DEF_FILE> Bazel configuration file for the commit SHA
# of the CMake <MODULE_NAME> module.
#
# Return parameters:
#   <OUT_TAG_SHA> - commit SHA of the dependency archive fetched from the Bazel configuration   
function(get_dependency_tag 
    MODULE_NAME 
    BAZEL_DEF_FILE
    OUT_TAG_SHA
)
    string(TOUPPER ${MODULE_NAME} MODULE_NAME_UPPERCASE)
    file(STRINGS ${BAZEL_DEF_FILE}
        LINE
        REGEX "^[ ]*${MODULE_NAME_UPPERCASE}_COMMIT[ ]*=[ ]*\"[a-zA-Z0-9]*\"$"
        LIMIT_COUNT 1)

  if(NOT LINE)
    print_parse_error(${MODULE_NAME} ${BAZEL_DEF_FILE} "${MODULE_NAME_UPPERCASE}_COMMIT")
  endif()

  extract_quote_contents("${LINE}" COMMIT_SHA)
  check_sha_length(${MODULE_NAME} ${BAZEL_DEF_FILE} "${MODULE_NAME_UPPERCASE}_COMMIT" ${COMMIT_SHA} 20)

  message(STATUS "Cloning ${MODULE_NAME} repository revision ${COMMIT_SHA}, found in ${BAZEL_DEF_FILE}")
  set(${OUT_TAG_SHA} ${COMMIT_SHA} PARENT_SCOPE)
endfunction()


# Parses the <BAZEL_DEF_FILE> Bazel configuration file for the archive
# URL of the CMake <MODULE_NAME> module.
#
# Return parameters:
#   <OUT_URL> - URL of the dependency archive fetched from the Bazel configuration
#   <OUT_CHECKSUM> - checksum of the archive from the Bazel configuration  
function(get_dependency_archive
    MODULE_NAME
    BAZEL_DEF_FILE
    OUT_URL
    OUT_CHECKSUM
)
    string(TOUPPER ${MODULE_NAME} MODULE_NAME_UPPERCASE)
    
    # Retrieve dependency archive URL
    file(STRINGS ${BAZEL_DEF_FILE}
        URL_LINE
        REGEX "^[ ]*${MODULE_NAME_UPPERCASE}_URL[ ]*=[ ]*\"http[a-zA-Z0-9/.:_-]*\"$"
        LIMIT_COUNT 1
    )
    if(NOT URL_LINE)
        print_parse_error(${MODULE_NAME} ${BAZEL_DEF_FILE} "${MODULE_NAME_UPPERCASE}_URL")
    endif()

    # Retrieve dependency archive checksum
    file(STRINGS ${BAZEL_DEF_FILE}
        CHECKSUM_LINE
        REGEX "^[ ]*${MODULE_NAME_UPPERCASE}_SHA256[ ]*=[ ]*\"[a-zA-Z0-9]*\"$"
        LIMIT_COUNT 1
    )

    if(NOT CHECKSUM_LINE)
        print_parse_error(${MODULE_NAME} ${BAZEL_DEF_FILE} "${MODULE_NAME_UPPERCASE}_SHA256")
    endif()

    extract_quote_contents(${URL_LINE} URL)
    extract_quote_contents(${CHECKSUM_LINE} CHECKSUM_SHA)

    check_sha_length(${MODULE_NAME} ${BAZEL_DEF_FILE} "${MODULE_NAME_UPPERCASE}_SHA256" ${CHECKSUM_SHA} 32)

    message(STATUS "Cloning ${MODULE_NAME} repository from ${URL}, found in ${BAZEL_DEF_FILE}")
    set(${OUT_URL} ${URL} PARENT_SCOPE)
    set(${OUT_CHECKSUM} ${CHECKSUM_SHA} PARENT_SCOPE)
endfunction()