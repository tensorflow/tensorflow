cmake_minimum_required(VERSION 2.8.3)

project(libjpeg)

set(LIBJPEG_SRCS
    "jaricom.c"
    "jcapimin.c"
    "jcapistd.c"
    "jcarith.c"
    "jccoefct.c"
    "jccolor.c"
    "jcdctmgr.c"
    "jchuff.c"
    "jcinit.c"
    "jcmainct.c"
    "jcmarker.c"
    "jcmaster.c"
    "jcomapi.c"
    "jcparam.c"
    "jcprepct.c"
    "jcsample.c"
    "jctrans.c"
    "jdapimin.c"
    "jdapistd.c"
    "jdarith.c"
    "jdatadst.c"
    "jdatasrc.c"
    "jdcoefct.c"
    "jdcolor.c"
    "jddctmgr.c"
    "jdhuff.c"
    "jdinput.c"
    "jdmainct.c"
    "jdmarker.c"
    "jdmaster.c"
    "jdmerge.c"
    "jdpostct.c"
    "jdsample.c"
    "jdtrans.c"
    "jerror.c"
    "jfdctflt.c"
    "jfdctfst.c"
    "jfdctint.c"
    "jidctflt.c"
    "jidctfst.c"
    "jidctint.c"
    "jmemmgr.c"
    "jmemnobs.c"
    "jquant1.c"
    "jquant2.c"
    "jutils.c"
)
set(LIBJPEG_INCLUDES
    "jconfig.h"
    "jdct.h"
    "jerror.h"
    "jinclude.h"
    "jmemsys.h"
    "jmorecfg.h"
    "jpegint.h"
    "jpeglib.h"
    "jversion.h"
)

if (WIN32)
  # Suppress warnings to reduce build log size.
  add_definitions(/wd4267 /wd4244 /wd4800 /wd4503 /wd4554 /wd4996 /wd4348 /wd4018)
  add_definitions(/wd4099 /wd4146 /wd4267 /wd4305 /wd4307)
  add_definitions(/wd4715 /wd4722 /wd4723 /wd4838 /wd4309 /wd4334)
  add_definitions(/wd4003 /wd4244 /wd4267 /wd4503 /wd4506 /wd4800 /wd4996)
  add_definitions(/wd8029)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(libjpeg ${LIBJPEG_SRCS})

install(TARGETS libjpeg
  RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(LIBJPEG_INCLUDE ${LIBJPEG_INCLUDES})
  install(FILES ${LIBJPEG_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
