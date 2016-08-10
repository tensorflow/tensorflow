include (ExternalProject)

set(gif_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/gif_archive)
set(gif_URL http://ufpr.dl.sourceforge.net/project/giflib/giflib-5.1.4.tar.gz)
set(gif_HASH SHA256=34a7377ba834397db019e8eb122e551a49c98f49df75ec3fcc92b9a794a4f6d1)
set(gif_INSTALL ${CMAKE_BINARY_DIR}/gif/install)
set(gif_STATIC_LIBRARIES ${gif_INSTALL}/lib/libgif.a)

set(gif_HEADERS
    "${gif_INSTALL}/include/gif_lib.h"
)

ExternalProject_Add(gif
    PREFIX gif
    URL ${gif_URL}
    URL_HASH ${gif_HASH}
    INSTALL_DIR ${gif_INSTALL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_COMMAND $(MAKE)
    INSTALL_COMMAND $(MAKE) install
    CONFIGURE_COMMAND
    ${CMAKE_CURRENT_BINARY_DIR}/gif/src/gif/configure
    --prefix=${gif_INSTALL}
    --enable-shared=yes
)

# put gif includes in the directory where they are expected
add_custom_target(gif_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${gif_INCLUDE_DIR}/giflib-5.1.4/lib
    DEPENDS gif)

add_custom_target(gif_copy_headers_to_destination
    DEPENDS gif_create_destination_dir)

foreach(header_file ${gif_HEADERS})
    add_custom_command(TARGET gif_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${header_file} ${gif_INCLUDE_DIR}/giflib-5.1.4/lib/)
endforeach()
