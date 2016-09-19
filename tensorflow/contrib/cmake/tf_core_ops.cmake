#def tf_gen_op_libs(op_lib_names):
#  # Make library out of each op so it can also be used to generate wrappers
#  # for various languages.
#  for n in op_lib_names:
#    native.cc_library(name=n + "_op_lib"
#                      copts=tf_copts(),
#                      srcs=["ops/" + n + ".cc"],
#                      deps=(["//tensorflow/core:framework"]),
#                      visibility=["//visibility:public"],
#                      alwayslink=1,
#                      linkstatic=1,)


set(tf_op_lib_names
    "array_ops"
    "attention_ops"
    "candidate_sampling_ops"
    "control_flow_ops"
    "ctc_ops"
    "data_flow_ops"
    "image_ops"
    "io_ops"
    "linalg_ops"
    "logging_ops"
    "functional_ops"
    "math_ops"
    "nn_ops"
    "no_op"
    "parsing_ops"
    "random_ops"
    "script_ops"
    "sendrecv_ops"
    "sparse_ops"
    "state_ops"
    "string_ops"
    "summary_ops"
    "training_ops"
)

foreach(tf_op_lib_name ${tf_op_lib_names})
    ########################################################
    # tf_${tf_op_lib_name} library
    ########################################################
    file(GLOB tf_${tf_op_lib_name}_srcs
        "${tensorflow_source_dir}/tensorflow/core/ops/${tf_op_lib_name}.cc"
    )

    add_library(tf_${tf_op_lib_name} OBJECT ${tf_${tf_op_lib_name}_srcs})

    add_dependencies(tf_${tf_op_lib_name} tf_core_framework)

    target_include_directories(tf_${tf_op_lib_name} PRIVATE
        ${tensorflow_source_dir}
        ${eigen_INCLUDE_DIRS}
    )

    target_compile_options(tf_${tf_op_lib_name} PRIVATE
        -fno-exceptions
        -DEIGEN_AVOID_STL_ARRAY
    )

    # C++11
    target_compile_features(tf_${tf_op_lib_name} PRIVATE
        cxx_rvalue_references
    )
endforeach()

#cc_library(
#    name = "user_ops_op_lib"
#    srcs = glob(["user_ops/**/*.cc"]),
#    copts = tf_copts(),
#    linkstatic = 1,
#    visibility = ["//visibility:public"],
#    deps = [":framework"],
#    alwayslink = 1,
#)
########################################################
# tf_user_ops library
########################################################
file(GLOB_RECURSE tf_user_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cc"
)

add_library(tf_user_ops OBJECT ${tf_user_ops_srcs})

add_dependencies(tf_user_ops tf_core_framework)

target_include_directories(tf_user_ops PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

target_compile_options(tf_user_ops PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_user_ops PRIVATE
    cxx_rvalue_references
)


#tf_cuda_library(
#    name = "ops"
#    srcs = glob(
#        [
#            "ops/**/*.h"
#            "ops/**/*.cc"
#            "user_ops/**/*.h"
#            "user_ops/**/*.cc"
#        ],
#        exclude = [
#            "**/*test*"
#            "**/*main.cc"
#            "user_ops/**/*.cu.cc"
#        ],
#    ),
#    copts = tf_copts(),
#    linkstatic = 1,
#    visibility = ["//visibility:public"],
#    deps = [
#        ":core"
#        ":lib"
#        ":protos_cc"
#        "//tensorflow/models/embedding:word2vec_ops"
#        "//third_party/eigen3"
#    ],
#    alwayslink = 1,
#)

########################################################
# tf_core_ops library
########################################################
file(GLOB_RECURSE tf_core_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/ops/*.h"
    "${tensorflow_source_dir}/tensorflow/core/ops/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.h"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cc"
)

file(GLOB_RECURSE tf_core_ops_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/core/ops/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/ops/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/ops/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cu.cc"
)

list(REMOVE_ITEM tf_core_ops_srcs ${tf_core_ops_exclude_srcs}) 

add_library(tf_core_ops OBJECT ${tf_core_ops_srcs})

add_dependencies(tf_core_ops tf_core_cpu)

target_include_directories(tf_core_ops PRIVATE
    ${tensorflow_source_dir}
    ${eigen_INCLUDE_DIRS}
)

#target_link_libraries(tf_core_ops
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${PROTOBUF_LIBRARIES}
#    tf_protos_cc
#    tf_core_lib
#    tf_core_cpu
#    tf_models_word2vec_ops
#)

target_compile_options(tf_core_ops PRIVATE
    -fno-exceptions
    -DEIGEN_AVOID_STL_ARRAY
)

# C++11
target_compile_features(tf_core_ops PRIVATE
    cxx_rvalue_references
)


