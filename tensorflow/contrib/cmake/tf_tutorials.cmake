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

target_link_libraries(tf_tutorials_example_trainer PUBLIC
    tf_protos_cc
    ${tensorflow_EXTERNAL_LIBRARIES}
)
