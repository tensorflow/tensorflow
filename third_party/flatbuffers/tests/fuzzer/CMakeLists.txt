cmake_minimum_required(VERSION 3.9)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(FlatBuffersFuzzerTests)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic -Werror -Wextra -Wno-unused-parameter -fsigned-char")

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -g -fsigned-char -fno-omit-frame-pointer")

# Typical slowdown introduced by MemorySanitizer (memory) is 3x.
# '-fsanitize=address' not allowed with '-fsanitize=memory'
if(YES)
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer,address,undefined")
else()
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer,memory,undefined -fsanitize-memory-track-origins=2")
endif()

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -fsanitize-coverage=edge,trace-cmp")

# enable link-time optimisation
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")

# https://llvm.org/docs/Passes.html
# save IR to see call graph
# make one bitcode file:> llvm-link *.bc -o out.bc
# print call-graph:> opt out.bc -analyze -print-callgraph &> callgraph.txt
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -flto")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")

set(FLATBUFFERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(FlatBuffers_Library_SRCS
  ${FLATBUFFERS_DIR}/include/flatbuffers/code_generators.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/base.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/flatbuffers.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/hash.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/idl.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/util.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/reflection.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/reflection_generated.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/stl_emulation.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/flexbuffers.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/registry.h
  ${FLATBUFFERS_DIR}/include/flatbuffers/minireflect.h
  ${FLATBUFFERS_DIR}/src/code_generators.cpp
  ${FLATBUFFERS_DIR}/src/idl_parser.cpp
  ${FLATBUFFERS_DIR}/src/idl_gen_text.cpp
  ${FLATBUFFERS_DIR}/src/reflection.cpp
  ${FLATBUFFERS_DIR}/src/util.cpp
  ${FLATBUFFERS_DIR}/tests/test_assert.cpp
)

include_directories(${FLATBUFFERS_DIR}/include)
include_directories(${FLATBUFFERS_DIR}/tests)
add_library(flatbuffers STATIC ${FlatBuffers_Library_SRCS})

# FLATBUFFERS_ASSERT should assert in Release as well.
# Redefine FLATBUFFERS_ASSERT macro definition.
# Declare as PUBLIC to cover asserts in all included header files.
target_compile_definitions(flatbuffers PUBLIC
   FLATBUFFERS_ASSERT=fuzzer_assert_impl)
target_compile_definitions(flatbuffers PUBLIC
   FLATBUFFERS_ASSERT_INCLUDE="${CMAKE_CURRENT_SOURCE_DIR}/fuzzer_assert.h")

if(NOT DEFINED FLATBUFFERS_MAX_PARSING_DEPTH)
  # Force checking of RecursionError in the test
  set(FLATBUFFERS_MAX_PARSING_DEPTH 8)
endif()
message(STATUS "FLATBUFFERS_MAX_PARSING_DEPTH: ${FLATBUFFERS_MAX_PARSING_DEPTH}")
target_compile_definitions(flatbuffers PRIVATE FLATBUFFERS_MAX_PARSING_DEPTH=8)

# Setup fuzzer tests.

add_executable(scalar_fuzzer flatbuffers_scalar_fuzzer.cc)
target_link_libraries(scalar_fuzzer PRIVATE flatbuffers)

add_executable(parser_fuzzer flatbuffers_parser_fuzzer.cc)
target_link_libraries(parser_fuzzer PRIVATE flatbuffers)

add_executable(verifier_fuzzer flatbuffers_verifier_fuzzer.cc)
target_link_libraries(verifier_fuzzer PRIVATE flatbuffers)
