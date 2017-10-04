#!/bin/bash -e
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# Builds double-conversion without bazel

set -e

# Make sure we're in the correct directory, at the root of the source tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../.."
cd "${ROOT_DIR}"

DOUBLE_CONVERSION_DIR="${ROOT_DIR}/tensorflow/contrib/makefile/downloads/double_conversion"
HOST_DOUBLE_CONVERSION_LIB=libdouble-conversion.a

MAKEFILE='
CXX := $(CC_PREFIX) gcc

OPTFLAGS := -O3
CXXFLAGS := $(OPTFLAGS)

INCLUDES := -I.

AR := ar
ARFLAGS := rc

OBJ_FILES := $(patsubst %.cc,%.o,$(wildcard double-conversion/*.cc))

LIB_PATH := '"${HOST_DOUBLE_CONVERSION_LIB}"'

all: $(LIB_PATH)

$(LIB_PATH) : $(OBJ_FILES)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(OBJ_FILES)

clean:
	rm $(OBJ_FILES) $(LIB_PATH)

.PHONY: clean
'

echo "${MAKEFILE}" > "${DOUBLE_CONVERSION_DIR}/Makefile"

cd "${DOUBLE_CONVERSION_DIR}"
make all --quiet
cd "${ROOT_DIR}"

echo "${DOUBLE_CONVERSION_DIR}/${HOST_DOUBLE_CONVERSION_LIB}"
