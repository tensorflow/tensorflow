#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# tensorflow-lite uses find_package for this package, so override the system
# installation and build from source instead.
include(flatbuffers)
if(flatbuffers_POPULATED)
  set(FLATBUFFERS_FOUND TRUE)
  get_target_property(FLATBUFFERS_INCLUDE_DIRS flatbuffers INCLUDE_DIRECTORIES)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
  set(FLATBUFFERS_LIBRARIES flatbuffers)
  set(FLATBUFFERS_PROJECT_DIR "${flatbuffers_SOURCE_DIR}" CACHE STRING
    "Flatbuffers project dir"
  )
endif()

