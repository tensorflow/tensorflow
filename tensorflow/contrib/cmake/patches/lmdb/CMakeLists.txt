cmake_minimum_required(VERSION 2.8.3)

project(liblmdb)

set(LIBLMDB_SRCS
    "libraries/liblmdb/mdb.c"
    "libraries/liblmdb/midl.c"
)

set(LIBLMDB_INCLUDES
    "libraries/liblmdb/lmdb.h"
    "libraries/liblmdb/midl.h"
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(lmdb ${LIBLMDB_SRCS})

install(TARGETS lmdb
  RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(LIBLMDB_INCLUDE ${LIBLMDB_INCLUDES})
  install(FILES ${LIBLMDB_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
