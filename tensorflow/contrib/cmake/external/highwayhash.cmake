include (ExternalProject)

set(highwayhash_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/highwayhash)
#set(highwayhash_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/highwayhash/src)
set(highwayhash_URL https://github.com/google/highwayhash.git)
set(highwayhash_TAG be5edafc2e1a455768e260ccd68ae7317b6690ee)
set(highwayhash_BUILD ${CMAKE_BINARY_DIR}/highwayhash/src/highwayhash)
set(highwayhash_INSTALL ${CMAKE_BINARY_DIR}/highwayhash/install)
#set(highwayhash_LIBRARIES ${highwayhash_BUILD}/obj/so/libhighwayhash.so)
set(highwayhash_STATIC_LIBRARIES
    ${highwayhash_INSTALL}/lib/libhighwayhash.a
)
set(highwayhash_INCLUDES ${highwayhash_BUILD})

set(highwayhash_HEADERS
    "${highwayhash_BUILD}/highwayhash/*.h"
)

ExternalProject_Add(highwayhash
    PREFIX highwayhash
    GIT_REPOSITORY ${highwayhash_URL}
    GIT_TAG ${highwayhash_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_COMMAND $(MAKE)
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
)

# put highwayhash includes in the directory where they are expected
add_custom_target(highwayhash_create_destination_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${highwayhash_INCLUDE_DIR}/highwayhash
    DEPENDS highwayhash)

add_custom_target(highwayhash_copy_headers_to_destination
    DEPENDS highwayhash_create_destination_dir)

foreach(header_file ${highwayhash_HEADERS})
    add_custom_command(TARGET highwayhash_copy_headers_to_destination PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${header_file} ${highwayhash_INCLUDE_DIR}/highwayhash)
endforeach()
