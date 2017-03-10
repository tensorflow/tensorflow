include (ExternalProject)

set(gemmlowp_URL http://github.com/google/gemmlowp/archive/a6f29d8ac48d63293f845f2253eccbf86bc28321.tar.gz)
set(gemmlowp_HASH SHA256=75d40ea8e68b0d1644f052fffe8f14a410b2a73d40ccb859a95c0578d194ec26)
set(gemmlowp_BUILD ${CMAKE_CURRENT_BINARY_DIR}/gemmlowp/src/gemmlowp)
set(gemmlowp_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gemmlowp/src/gemmlowp)

ExternalProject_Add(gemmlowp
    PREFIX gemmlowp
    URL ${gemmlowp_URL}
    URL_HASH ${gemmlowp_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/patches/gemmlowp/CMakeLists.txt ${gemmlowp_BUILD}
    INSTALL_COMMAND "")
