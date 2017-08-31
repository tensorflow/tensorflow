cmake_minimum_required(VERSION 2.8.3)

project(farmhash)

set(FARMHASH_SRCS
    "src/farmhash.h"
    "src/farmhash.cc"
)

set(FARMHASH_INCLUDES
    "src/farmhash.h"
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(farmhash ${FARMHASH_SRCS})
add_definitions(-DFARMHASH_NO_BUILTIN_EXPECT)

install(TARGETS farmhash
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(FARMHASH_INCLUDE ${FARMHASH_INCLUDES})
  install(FILES ${FARMHASH_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
