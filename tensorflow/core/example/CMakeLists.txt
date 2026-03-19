#
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

if (NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf REQUIRED)
endif()

set(GEN_PROTO_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/example)

# Generate feature proto .h, .cc and lib.
list(APPEND feature_generated_files ${GEN_PROTO_DIR}/feature.pb.h ${GEN_PROTO_DIR}/feature.pb.cc)

add_custom_command(
  OUTPUT ${feature_generated_files}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
  ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${TENSORFLOW_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/feature.proto
  DEPENDS ${Protobuf_PROTOC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/feature.proto
)

set_source_files_properties(${feature_generated_files} PROPERTIES GENERATED TRUE)
add_library(feature_proto ${feature_generated_files})
target_link_libraries(feature_proto protobuf::libprotobuf)
target_include_directories(feature_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Generate example proto .h, .cc and lib.
list(APPEND example_generated_files ${GEN_PROTO_DIR}/example.pb.h ${GEN_PROTO_DIR}/example.pb.cc)

add_custom_command(
  OUTPUT ${example_generated_files}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
  ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${TENSORFLOW_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/example.proto
  DEPENDS ${Protobuf_PROTOC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/example.proto ${feature_generated_files}
)

set_source_files_properties(${example_generated_files} PROPERTIES GENERATED TRUE)
add_library(example_proto ${example_generated_files})
target_link_libraries(example_proto feature_proto protobuf::libprotobuf)
target_include_directories(example_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})