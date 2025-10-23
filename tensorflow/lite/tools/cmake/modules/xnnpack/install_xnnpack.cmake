# Patch: Install XNNPACK library and headers for TensorFlow Lite

install(TARGETS XNNPACK
        EXPORT tensorflow-liteTargets
        DESTINATION lib)

install(DIRECTORY ${xnnpack_SOURCE_DIR}/include/
        DESTINATION include)
