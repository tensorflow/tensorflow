#new_http_archive(
#  name = "eigen_archive",
#  url = "https://bitbucket.org/eigen/eigen/get/...",
#  sha256 = "...",
#  build_file = "eigen.BUILD",
#)

include (ExternalProject)

# We parse the current Eigen version and archive hash from the bazel configuration
file(STRINGS ${PROJECT_SOURCE_DIR}/../../workspace.bzl workspace_contents)
foreach(line ${workspace_contents})
    string(REGEX MATCH ".*eigen_version.*=.*\"(.*)\"" has_version ${line})
    if(has_version)
        set(eigen_version ${CMAKE_MATCH_1})
        break()
    endif()
endforeach()
foreach(line ${workspace_contents})
    string(REGEX MATCH ".*eigen_sha256.*=.*\"(.*)\"" has_hash ${line})
    if(has_hash)
        set(eigen_hash ${CMAKE_MATCH_1})
        break()
    endif()
endforeach()

set(eigen_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/external/eigen_archive
    ${tensorflow_source_dir}/third_party/eigen3
)
set(eigen_URL https://bitbucket.org/eigen/eigen/get/${eigen_version}.tar.gz)
set(eigen_HASH SHA256=${eigen_hash})
set(eigen_BUILD ${CMAKE_CURRENT_BINARY_DIR}/eigen/src/eigen)
set(eigen_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/eigen/install)

ExternalProject_Add(eigen
    PREFIX eigen
    URL ${eigen_URL}
    URL_HASH ${eigen_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    INSTALL_DIR "${eigen_INSTALL}"
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -DCMAKE_INSTALL_PREFIX:STRING=${eigen_INSTALL}
        -DINCLUDE_INSTALL_DIR:STRING=${CMAKE_CURRENT_BINARY_DIR}/external/eigen_archive
)
