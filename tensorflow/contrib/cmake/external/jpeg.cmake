include (ExternalProject)

set(jpeg_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/jpeg_archive)
set(jpeg_URL http://www.ijg.org/files/jpegsrc.v9a.tar.gz)
set(jpeg_HASH SHA256=3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7)
set(jpeg_BUILD ${CMAKE_CURRENT_BINARY_DIR}/jpeg/src/jpeg)
set(jpeg_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/jpeg/install)

if(WIN32)
  set(jpeg_STATIC_LIBRARIES ${jpeg_INSTALL}/lib/libjpeg.lib)
else()
  set(jpeg_STATIC_LIBRARIES ${jpeg_INSTALL}/lib/libjpeg.a)
endif()

set(jpeg_HEADERS
    "${jpeg_INSTALL}/include/jconfig.h"
    "${jpeg_INSTALL}/include/jerror.h"
    "${jpeg_INSTALL}/include/jmorecfg.h"
    "${jpeg_INSTALL}/include/jpeglib.h"
    "${jpeg_BUILD}/cderror.h"
    "${jpeg_BUILD}/cdjpeg.h"
    "${jpeg_BUILD}/jdct.h"
    "${jpeg_BUILD}/jinclude.h"
    "${jpeg_BUILD}/jmemsys.h"
    "${jpeg_BUILD}/jpegint.h"
    "${jpeg_BUILD}/jversion.h"
    "${jpeg_BUILD}/transupp.h"
)

if (WIN32)
    ExternalProject_Add(jpeg
        PREFIX jpeg
        URL ${jpeg_URL}
        URL_HASH ${jpeg_HASH}
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/jpeg/CMakeLists.txt ${jpeg_BUILD}
        INSTALL_DIR ${jpeg_INSTALL}
        DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
        CMAKE_CACHE_ARGS
            -DCMAKE_BUILD_TYPE:STRING=Release
            -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
            -DCMAKE_INSTALL_PREFIX:STRING=${jpeg_INSTALL}
    )

    ExternalProject_Add_Step(jpeg copy_jconfig
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${jpeg_BUILD}/jconfig.vc ${jpeg_BUILD}/jconfig.h
        DEPENDEES patch
        DEPENDERS build
    )

else()
    ExternalProject_Add(jpeg
        PREFIX jpeg
        URL ${jpeg_URL}
        URL_HASH ${jpeg_HASH}
        INSTALL_DIR ${jpeg_INSTALL}
        DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        CONFIGURE_COMMAND
            ${jpeg_BUILD}/configure
            --prefix=${jpeg_INSTALL}
            --enable-shared=yes
	    CFLAGS=-fPIC
    )
  
endif()

# put jpeg includes in the directory where they are expected
add_custom_target(jpeg_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${jpeg_INCLUDE_DIR}
    DEPENDS jpeg)

add_custom_target(jpeg_copy_headers_to_destination
    DEPENDS jpeg_create_destination_dir)

foreach(header_file ${jpeg_HEADERS})
    add_custom_command(TARGET jpeg_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header_file} ${jpeg_INCLUDE_DIR})
endforeach()
