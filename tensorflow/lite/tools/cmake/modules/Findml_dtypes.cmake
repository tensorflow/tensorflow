#
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

include(ml_dtypes)
if(ml_dtypes_POPULATED)
  get_target_property(ML_DTYPES_INCLUDE_DIRS ml_dtypes INTERFACE_DIRECTORIES)
endif()

if(ml_dtypes_FOUND OR ml_dtypes_POPULATED)
  set(ML_DTYPES_FOUND TRUE)
  add_library(ml_dtypes::ml_dtypes ALIAS ml_dtypes)
  set(ML_DTYPES_LIBRARIES ml_dtypes::ml_dtypes)
endif()
