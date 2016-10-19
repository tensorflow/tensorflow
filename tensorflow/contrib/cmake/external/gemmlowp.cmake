include (ExternalProject)

set(gemmlowp_URL http://github.com/google/gemmlowp/archive/c0bacf11fb509a2cbe15a97362a2df067ffd57a2.tar.gz)
set(gemmlowp_HASH SHA256=dc64a38f9927db18748d9024987c9b102115e25bc2be4b76aa8e422b8f83d882)
set(gemmlowp_BUILD ${CMAKE_BINARY_DIR}/gemmlowp/src/gemmlowp)
set(gemmlowp_INCLUDE_DIR ${CMAKE_BINARY_DIR}/gemmlowp/src/gemmlowp)

ExternalProject_Add(gemmlowp
    PREFIX gemmlowp
    URL ${gemmlowp_URL}
    URL_HASH ${gemmlowp_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/patches/gemmlowp/CMakeLists.txt ${gemmlowp_BUILD}
    INSTALL_COMMAND "")
