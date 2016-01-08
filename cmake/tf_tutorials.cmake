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
)

#target_include_directories(tf_tutorials_example_trainer PUBLIC
#    ${tensorflow_source_dir}/third_party/eigen3
#)

target_link_libraries(tf_tutorials_example_trainer PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    ${PROTOBUF_LIBRARIES}
    tf_protos_cc
    tf_core_lib
    tf_core_cpu
    tf_core_framework
    tf_core_kernels
    tf_core_direct_session
    tf_cc_ops
)

target_compile_options(tf_tutorials_example_trainer PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_tutorials_example_trainer PRIVATE
    cxx_rvalue_references
)
