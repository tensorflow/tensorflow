#new_http_archive(
#  name = "eigen_archive",
#  urls = ["https://bitbucket.org/eigen/eigen/get/..."],
#  sha256 = "...",
#  build_file = "eigen.BUILD",
#)

include (ExternalProject)

# We parse the current Eigen version and archive hash from the bazel configuration
file(STRINGS ${PROJECT_SOURCE_DIR}/../../workspace.bzl workspace_contents)
foreach(line ${workspace_contents})
    string(REGEX MATCH ".*\"(https://bitbucket.org/eigen/eigen/get/[^\"]*tar.gz)\"" has_url ${line})
    if(has_url)
        set(eigen_url ${CMAKE_MATCH_1})
        break()
    endif()
endforeach()

set(eigen_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/external/eigen_archive
    ${tensorflow_source_dir}/third_party/eigen3
)
set(eigen_URL ${eigen_url})
set(eigen_BUILD ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen)
set(eigen_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/eigen/install)

ExternalProject_Add(eigen
    PREFIX eigen
    URL ${eigen_URL}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    INSTALL_DIR "${eigen_INSTALL}"
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:STRING=${eigen_INSTALL}
        -DINCLUDE_INSTALL_DIR:STRING=${CMAKE_CURRENT_BINARY_DIR}/external/eigen_archive
)
