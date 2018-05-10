# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
        "${tensorflow_source_dir}/tensorflow/compiler/xla/statusor.h"
        "${tensorflow_source_dir}/tensorflow/compiler/xla/statusor.cc"
    )
    if (NOT tensorflow_BUILD_CC_TESTS)
        file(GLOB tf_stream_executor_gpu_tests
            "${tensorflow_source_dir}/tensorflow/stream_executor/cuda/*_test.cc"
        )
        list(REMOVE_ITEM tf_stream_executor_gpu_srcs ${tf_stream_executor_gpu_tests})
    endif()
    list(APPEND tf_stream_executor_srcs ${tf_stream_executor_gpu_srcs})
endif()

#file(GLOB_RECURSE tf_stream_executor_test_srcs
#    "${tensorflow_source_dir}/tensorflow/stream_executor/*_test.cc"
#    "${tensorflow_source_dir}/tensorflow/stream_executor/*_test.h"
#)
#list(REMOVE_ITEM tf_stream_executor_srcs ${tf_stream_executor_test_srcs})

if (NOT WIN32)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgomp")
endif (NOT WIN32)
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
