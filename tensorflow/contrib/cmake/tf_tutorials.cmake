#cc_binary(
#    name = "tutorials_example_trainer",
#    srcs = ["tutorials/example_trainer.cc"],
#    copts = tf_copts(),
#    linkopts = [
#        "-lpthread",
#        "-lm",
#    ],
#    deps = [
#        ":cc_ops",
#        "//tensorflow/core:kernels",
#        "//tensorflow/core:tensorflow",
#    ],
#)

set(tf_tutorials_example_trainer_srcs
    "${tensorflow_source_dir}/tensorflow/cc/tutorials/example_trainer.cc"
)

add_executable(tf_tutorials_example_trainer
    ${tf_tutorials_example_trainer_srcs}
    $<TARGET_OBJECTS:tf_core_lib>
    $<TARGET_OBJECTS:tf_core_cpu>
    $<TARGET_OBJECTS:tf_core_framework>
    $<TARGET_OBJECTS:tf_core_kernels>
    $<TARGET_OBJECTS:tf_cc_framework>
    $<TARGET_OBJECTS:tf_cc_ops>
    $<TARGET_OBJECTS:tf_core_ops>
    $<TARGET_OBJECTS:tf_core_direct_session>
)

target_include_directories(tf_tutorials_example_trainer PUBLIC
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

target_link_libraries(tf_tutorials_example_trainer PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROTOBUF_STATIC_LIBRARIES}
    tf_protos_cc
    ${boringssl_STATIC_LIBRARIES}
    ${farmhash_STATIC_LIBRARIES}
    ${gif_STATIC_LIBRARIES}
    ${jpeg_STATIC_LIBRARIES}
    ${jsoncpp_STATIC_LIBRARIES}
    ${png_STATIC_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

target_compile_options(tf_tutorials_example_trainer PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_tutorials_example_trainer PRIVATE
    cxx_rvalue_references
)
