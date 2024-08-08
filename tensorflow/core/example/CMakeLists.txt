
if (NOT TARGET protobuf::libprotobuf)
  find_package(Protobuf REQUIRED)
endif()

# CMAKE_CURRENT_BINARY_DIR is the binary path set in the add_subdirectory in the
# top-level CMakeLists.txt.
set(GEN_PROTO_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/core/example)

# Generate foo proto .h, .cc and lib.
list(APPEND foo_generated_files ${GEN_PROTO_DIR}/foo.pb.h ${GEN_PROTO_DIR}/foo.pb.cc)
add_custom_command(
  OUTPUT ${foo_generated_files}
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
  ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${TENSORFLOW_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/foo.proto
  DEPENDS ${Protobuf_PROTOC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/foo.proto
)
set_source_files_properties(${foo_generated_files} PROPERTIES GENERATED TRUE)
add_library(foo_proto ${foo_generated_files})
target_link_libraries(foo_proto protobuf::libprotobuf)
target_include_directories(foo_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
