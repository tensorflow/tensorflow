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

# tensorflow-lite uses find_package for this package, so build from
# source if the system version is not enabled.

if(SYSTEM_FARMHASH)
  include(FindPackageHandleStandardArgs)
  find_path(FARMHASH_ROOT_DIR NAMES include/farmhash.h)
  find_library(FARMHASH_LIB NAMES farmhash PATHS ${FARMHASH_ROOT_DIR}/lib ${FARMHASH_LIB_PATH})
  find_path(FARMHASH_INCLUDE_DIRS NAMES farmhash.h PATHS ${FARMHASH_ROOT_DIR}/include)
  find_package_handle_standard_args(farmhash DEFAULT_MSG FARMHASH_LIB FARMHASH_INCLUDE_DIRS)
endif()

if(farmhash_FOUND)
  add_library(farmhash SHARED IMPORTED GLOBAL)
  set_target_properties(farmhash PROPERTIES
    IMPORTED_LOCATION ${FARMHASH_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${FARMHASH_INCLUDE_DIRS}
  )
else()
  include(farmhash)
  if(farmhash_POPULATED)
    get_target_property(FARMHASH_INCLUDE_DIRS farmhash INTERFACE_DIRECTORIES)
  endif()
endif()

if(farmhash_FOUND OR farmhash_POPULATED)
  set(FARMHASH_FOUND TRUE)
  add_library(farmhash::farmhash ALIAS farmhash)
  set(FARMHASH_LIBRARIES farmhash::farmhash)
endif()

