set(tf_c_srcs
    "${tensorflow_source_dir}/tensorflow/c/c_api.cc"
    "${tensorflow_source_dir}/tensorflow/c/c_api.h"
    "${tensorflow_source_dir}/tensorflow/c/checkpoint_reader.cc"
    "${tensorflow_source_dir}/tensorflow/c/checkpoint_reader.h"
    "${tensorflow_source_dir}/tensorflow/c/tf_status_helper.cc"
    "${tensorflow_source_dir}/tensorflow/c/tf_status_helper.h"
)

add_library(tf_c OBJECT ${tf_c_srcs})
add_dependencies(tf_c tf_core_lib tf_protos_cc)
