# ParseExample compile database coverage for CMake builds.

add_library(tflite_parse_example_compiledb OBJECT EXCLUDE_FROM_ALL
  example_proto_fast_parsing.cc
  parse_example.cc
  parse_example_test.cc
)

target_link_libraries(tflite_parse_example_compiledb
  PRIVATE
    example_proto
    tensorflow-lite
)

if(TARGET gmock)
  target_link_libraries(tflite_parse_example_compiledb PRIVATE gmock)
endif()