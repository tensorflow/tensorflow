if (NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf REQUIRED)
endif()

# CMAKE_CURRENT_BINARY_DIR is the binary path set in the add_subdirectory in the
# top-level CMakeLists.txt.
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

# We need to set proto_path to TENSORFLOW_SOURCE_DIR, because example.proto 
# import "tensorflow/core/example/feature.proto". We need to maintain the
# relative path.
# After we set proto_path to TENSORFLOW_SOURCE_DIR. There are two artifacts:
# (1) The example.pb.h/cc are generated under cpp_out/tensorflow/core/example/.
# (2) The generated example.pb.cc will include 
# "tensorflow/core/example/example.pb.h", example.pb.h will include 
# "tensorflow/core/example/feature.pb.h". Therefore, when we build the proto 
# library, we need to set include directories to the same cpp_out directory,
# which is CMAKE_CURRENT_BINARY_DIR here. We also set proto_path and cpp_out for
# feature.proto to be the same as example.proto, so that all the generated files
# are under the same folder hierarchy.
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