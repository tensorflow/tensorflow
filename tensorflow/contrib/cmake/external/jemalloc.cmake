include (ExternalProject)

set(jemalloc_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc/include)
set(jemalloc_URL https://github.com/jemalloc/jemalloc-cmake/archive/jemalloc-cmake.4.3.1.tar.gz)
set(jemalloc_HASH SHA256=f9be9a05fe906deb5c1c8ca818071a7d2e27d66fd87f5ba9a7bf3750bcedeaf0)
set(jemalloc_BUILD ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc)

if (WIN32)
    set(jemalloc_INCLUDE_DIRS
        ${jemalloc_INCLUDE_DIRS} 
        ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc/include/msvc_compat
    )
    set(jemalloc_ADDITIONAL_CMAKE_OPTIONS -A x64)
    set(jemalloc_STATIC_LIBRARIES ${jemalloc_BUILD}/Release/jemalloc.lib)
else()
    set(jemalloc_STATIC_LIBRARIES ${jemalloc_BUILD}/Release/jemalloc.a)
endif()

ExternalProject_Add(jemalloc
    PREFIX jemalloc
    URL ${jemalloc_URL}
    URL_HASH ${jemalloc_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -Dwith-jemalloc-prefix:STRING=jemalloc_
        -Dwithout-export:BOOL=ON
        ${jemalloc_ADDITIONAL_CMAKE_OPTIONS}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target jemalloc
    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping install step."
)
