include (ExternalProject)

set(gemmlowp_URL http://github.com/google/gemmlowp/archive/18b0aab27eaa5c009f27692afef89ef200181fbc.tar.gz)
set(gemmlowp_HASH SHA256=5a13a90b33d0359a7c027d258f9848ff0f4499ac9858a0fd9d47d7fbf7364513)
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
