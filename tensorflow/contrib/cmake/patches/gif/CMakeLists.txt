cmake_minimum_required(VERSION 2.8.3)

project(giflib)

set(GIFLIB_SRCS
    "lib/dgif_lib.c"
    "lib/egif_lib.c"
    "lib/gif_font.c"
    "lib/gif_hash.h"
    "lib/gifalloc.c"
    "lib/openbsd-reallocarray.c"
    "lib/gif_err.c"
    "lib/quantize.c"
    "lib/gif_hash.c"
    "lib/gif_lib.h"
    "lib/gif_lib_private.h"
)
set(GIFLIB_INCLUDES
    "lib/gif_lib.h"
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")

add_library(giflib ${GIFLIB_SRCS})

install(TARGETS giflib
  RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(GIFLIB_INCLUDE ${GIFLIB_INCLUDES})
  install(FILES ${GIFLIB_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
