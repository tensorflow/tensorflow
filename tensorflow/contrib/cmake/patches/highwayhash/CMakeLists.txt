cmake_minimum_required(VERSION 2.8.3)

project(highwayhash)

set(HIGHWAYHASH_SRCS
    "highwayhash/code_annotation.h"
    "highwayhash/highway_tree_hash.cc"
    "highwayhash/highway_tree_hash.h"
    "highwayhash/scalar_highway_tree_hash.cc"
    "highwayhash/scalar_highway_tree_hash.h"
    "highwayhash/scalar_sip_tree_hash.cc"
    "highwayhash/scalar_sip_tree_hash.h"
    "highwayhash/sip_hash.cc"
    "highwayhash/sip_hash.h"
    "highwayhash/sip_tree_hash.cc"
    "highwayhash/sip_tree_hash.h"
    "highwayhash/sse41_highway_tree_hash.cc"
    "highwayhash/sse41_highway_tree_hash.h"
    "highwayhash/state_helpers.h"
    "highwayhash/types.h"
    "highwayhash/vec.h"
    "highwayhash/vec2.h"
)

set(HIGHWAYHASH_INCLUDES
    "highwayhash/code_annotation.h"
    "highwayhash/highway_tree_hash.h"
    "highwayhash/scalar_highway_tree_hash.h"
    "highwayhash/scalar_sip_tree_hash.h"
    "highwayhash/sip_hash.h"
    "highwayhash/sip_tree_hash.h"
    "highwayhash/sse41_highway_tree_hash.h"
    "highwayhash/state_helpers.h"
    "highwayhash/types.h"
    "highwayhash/vec.h"
    "highwayhash/vec2.h"
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(highwayhash ${HIGHWAYHASH_SRCS})

# C++11
target_compile_features(highwayhash PRIVATE
    cxx_rvalue_references
)

install(TARGETS highwayhash
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(HIGHWAYHASH_INCLUDE ${HIGHWAYHASH_INCLUDES})
  install(FILES ${HIGHWAYHASH_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
