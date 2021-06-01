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
include(eigen)
if(eigen_POPULATED)
  set(EIGEN_FOUND TRUE)
  get_target_property(EIGEN_INCLUDE_DIRS eigen INTERFACE_DIRECTORIES)
  # If using MSVC2015 (14) or below force Eigen to C++11.
  if(MSVC_VERSION LESS_EQUAL 1900)
    set(COMPILE_DEFINITIONS "EIGEN_MAX_CPP_VER=11")
    get_target_property(EIGEN_COMPILE_DEFINITIONS eigen
      INTERFACE_COMPILE_DEFINITIONS
    )
    if(EIGEN_COMPILE_DEFINITIONS)
      list(APPEND COMPILE_DEFINITIONS ${EIGEN_COMPILE_DEFINITIONS})
    endif()
    set_property(
      TARGET
        eigen
      PROPERTY
        INTERFACE_COMPILE_DEFINITIONS
        ${COMPILE_DEFINITIONS}
    )
  endif()
  set(EIGEN_LIBRARIES Eigen3::Eigen)
endif()

