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

if (WIN32)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_definitions(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")

add_library(giflib ${GIFLIB_SRCS})

install(TARGETS giflib
  RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(GIFLIB_INCLUDE ${GIFLIB_INCLUDES})
  install(FILES ${GIFLIB_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
