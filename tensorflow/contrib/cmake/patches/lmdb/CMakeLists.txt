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

if (WIN32)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_definitions(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(lmdb ${LIBLMDB_SRCS})

install(TARGETS lmdb
  RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(LIBLMDB_INCLUDE ${LIBLMDB_INCLUDES})
  install(FILES ${LIBLMDB_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
