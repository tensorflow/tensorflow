include (ExternalProject)

set(png_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/png_archive)
set(png_URL https://storage.googleapis.com/libpng-public-archive/libpng-1.2.53.tar.gz)
set(png_HASH SHA256=e05c9056d7f323088fd7824d8c6acc03a4a758c4b4916715924edc5dd3223a72)
set(png_BUILD ${CMAKE_BINARY_DIR}/png/src/png)
set(png_INSTALL ${CMAKE_BINARY_DIR}/png/install)

if(WIN32)
  set(png_STATIC_LIBRARIES ${CMAKE_BINARY_DIR}/png/install/lib/libpng12_static.lib)
else()
  set(png_STATIC_LIBRARIES ${CMAKE_BINARY_DIR}/png/install/lib/libpng12.a)
endif()

set(png_HEADERS
    "${png_INSTALL}/include/libpng12/png.h"
    "${png_INSTALL}/include/libpng12/pngconf.h"
)

ExternalProject_Add(png
    PREFIX png
    DEPENDS zlib
    URL ${png_URL}
    URL_HASH ${png_HASH}
    INSTALL_DIR ${png_INSTALL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:STRING=${png_INSTALL}
	-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
	-DZLIB_ROOT:STRING=${ZLIB_INSTALL}
)

## put png includes in the directory where they are expected
add_custom_target(png_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${png_INCLUDE_DIR}
    DEPENDS png)

add_custom_target(png_copy_headers_to_destination
    DEPENDS png_create_destination_dir)

foreach(header_file ${png_HEADERS})
  add_custom_command(TARGET png_copy_headers_to_destination PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${header_file} ${png_INCLUDE_DIR}/)
endforeach()
