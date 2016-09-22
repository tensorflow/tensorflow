include (ExternalProject)

set(boringssl_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/boringssl/src/boringssl/include)
#set(boringssl_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/boringssl/src)
set(boringssl_URL https://boringssl.googlesource.com/boringssl)
set(boringssl_TAG e72df93)
set(boringssl_BUILD ${CMAKE_BINARY_DIR}/boringssl/src/boringssl-build)
#set(boringssl_LIBRARIES ${boringssl_BUILD}/obj/so/libboringssl.so)
set(boringssl_STATIC_LIBRARIES
    ${boringssl_BUILD}/ssl/libssl.a
    ${boringssl_BUILD}/crypto/libcrypto.a
    ${boringssl_BUILD}/decrepit/libdecrepit.a
)
set(boringssl_INCLUDES ${boringssl_BUILD})

set(boringssl_HEADERS
    "${boringssl_INCLUDE_DIR}/include/*.h"
)

ExternalProject_Add(boringssl
    PREFIX boringssl
    GIT_REPOSITORY ${boringssl_URL}
    GIT_TAG ${boringssl_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    # BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
)

