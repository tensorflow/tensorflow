# Patch: Install pthreadpool library and headers for TensorFlow Lite

install(TARGETS pthreadpool
        EXPORT tensorflow-liteTargets
        DESTINATION lib)

install(DIRECTORY ${pthreadpool_SOURCE_DIR}/include/
        DESTINATION include)
