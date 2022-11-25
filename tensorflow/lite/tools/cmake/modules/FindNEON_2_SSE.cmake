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
include(neon2sse)
if(neon2sse_POPULATED)
  set(NEON2SSE_FOUND TRUE)
  get_target_property(NEON2SSE_INCLUDE_DIRS NEON_2_SSE INTERFACE_DIRECTORIES)
  add_library(NEON_2_SSE::NEON_2_SSE ALIAS NEON_2_SSE)
  set(NEON2SSE_LIBRARIES NEON_2_SSE)
endif()
