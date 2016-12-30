########################################################
# tf_core_cpu library
########################################################
file(GLOB_RECURSE tf_core_cpu_srcs
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/*.h"
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/*.h"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/graph/*.h"
    "${tensorflow_source_dir}/tensorflow/core/graph/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/public/*.h"
)

file(GLOB_RECURSE tf_core_cpu_exclude_srcs
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/*test*.h"
    "${tensorflow_source_dir}/tensorflow/cc/saved_model/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/*test*.h"
    "${tensorflow_source_dir}/tensorflow/core/*test*.cc"
    "${tensorflow_source_dir}/tensorflow/core/*main.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu_device_factory.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/direct_session.h"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_factory.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/session_options.cc"
)
list(REMOVE_ITEM tf_core_cpu_srcs ${tf_core_cpu_exclude_srcs}) 

# We need to include stubs for the GPU tracer, which are in the exclude glob.
list(APPEND tf_core_cpu_srcs
     "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_tracer.cc"
     "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/gpu_tracer.h"
)

if (tensorflow_ENABLE_GPU)
  file(GLOB_RECURSE tf_core_gpu_srcs
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu/*.cc"
    "${tensorflow_source_dir}/tensorflow/core/platform/default/gpu/cupti_wrapper.cc"
    "${tensorflow_source_dir}/tensorflow/core/common_runtime/gpu_device_factory.cc"
  )
  file(GLOB_RECURSE tf_core_gpu_exclude_srcs
     "${tensorflow_source_dir}/tensorflow/core/*test*.cc"
     "${tensorflow_source_dir}/tensorflow/core/*test*.cc"
  )
  list(REMOVE_ITEM tf_core_gpu_srcs ${tf_core_gpu_exclude_srcs})
  list(APPEND tf_core_cpu_srcs ${tf_core_gpu_srcs})
endif()

add_library(tf_core_cpu OBJECT ${tf_core_cpu_srcs})
add_dependencies(tf_core_cpu tf_core_framework)
