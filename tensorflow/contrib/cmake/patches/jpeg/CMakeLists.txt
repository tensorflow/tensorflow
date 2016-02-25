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

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_library(libjpeg ${LIBJPEG_SRCS})

install(TARGETS libjpeg
  RUNTIME DESTINATION bin COMPONENT RuntimeLibraries
  LIBRARY DESTINATION lib COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION lib COMPONENT Development)

foreach(LIBJPEG_INCLUDE ${LIBJPEG_INCLUDES})
  install(FILES ${LIBJPEG_INCLUDE} DESTINATION include COMPONENT Development)
endforeach()
