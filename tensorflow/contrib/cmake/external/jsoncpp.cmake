include (ExternalProject)

set(jsoncpp_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp/src/jsoncpp)
#set(jsoncpp_EXTRA_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp/src)
set(jsoncpp_URL https://github.com/open-source-parsers/jsoncpp.git)
set(jsoncpp_TAG 4356d9b)
set(jsoncpp_BUILD ${CMAKE_BINARY_DIR}/jsoncpp/src/jsoncpp/src/lib_json)
set(jsoncpp_LIBRARIES ${jsoncpp_BUILD}/obj/so/libjsoncpp.so)
get_filename_component(jsoncpp_STATIC_LIBRARIES ${jsoncpp_BUILD}/libjsoncpp.a ABSOLUTE)
set(jsoncpp_INCLUDES ${jsoncpp_BUILD})

# We only need jsoncpp.h in external/jsoncpp/jsoncpp/jsoncpp.h
# For the rest, we'll just add the build dir as an include dir.
set(jsoncpp_HEADERS
    "${jsoncpp_INCLUDE_DIR}/include/json/json.h"
)

ExternalProject_Add(jsoncpp
    PREFIX jsoncpp
    GIT_REPOSITORY ${jsoncpp_URL}
    GIT_TAG ${jsoncpp_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
)

