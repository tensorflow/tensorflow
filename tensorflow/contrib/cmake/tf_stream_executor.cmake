#cc_library(
#    name = "stream_executor",
#    srcs = glob(
#        [
#XX            "*.cc",
#            "lib/*.cc",
#        ],
#        exclude = [
#            "**/*_test.cc",
#        ],
#    ) + if_cuda(
#        glob([
#            "cuda/*.cc",
#        ]),
#    ),
#    hdrs = glob([
#        "*.h",
#        "cuda/*.h",
#        "lib/*.h",
#        "platform/**/*.h",
#    ]),
#    data = [
#        "//tensorflow/core:cuda",
#        "//third_party/gpus/cuda:cublas",
#        "//third_party/gpus/cuda:cudnn",
#    ],
#    linkopts = [
#        "-ldl",
#    ],
#    visibility = ["//visibility:public"],
#    deps = [
#        "//tensorflow/core:lib",
#        "//third_party/gpus/cuda:cuda_headers",
#    ],
#    alwayslink = 1,
#)

########################################################
# tf_stream_executor library
########################################################
file(GLOB tf_stream_executor_srcs
    "${tensorflow_source_dir}/tensorflow/stream_executor/*.cc"
    "${tensorflow_source_dir}/tensorflow/stream_executor/*.h"
    "${tensorflow_source_dir}/tensorflow/stream_executor/lib/*.cc"
    "${tensorflow_source_dir}/tensorflow/stream_executor/lib/*.h"
    "${tensorflow_source_dir}/tensorflow/stream_executor/platform/*.h"
    "${tensorflow_source_dir}/tensorflow/stream_executor/platform/default/*.h"
)

if (tensorflow_ENABLE_GPU)    
    file(GLOB tf_stream_executor_gpu_srcs
        "${tensorflow_source_dir}/tensorflow/stream_executor/cuda/*.cc"
    )
    list(APPEND tf_stream_executor_srcs ${tf_stream_executor_gpu_srcs})
endif()    

#file(GLOB_RECURSE tf_stream_executor_test_srcs
#    "${tensorflow_source_dir}/tensorflow/stream_executor/*_test.cc"
#    "${tensorflow_source_dir}/tensorflow/stream_executor/*_test.h"
#)
#list(REMOVE_ITEM tf_stream_executor_srcs ${tf_stream_executor_test_srcs}) 

add_library(tf_stream_executor OBJECT ${tf_stream_executor_srcs})

add_dependencies(tf_stream_executor
    tf_core_lib
)
#target_link_libraries(tf_stream_executor
#    ${CMAKE_THREAD_LIBS_INIT}
#    ${PROTOBUF_LIBRARIES}
#    tf_protos_cc
#    tf_core_lib
#)
