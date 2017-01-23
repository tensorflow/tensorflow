########################################################
# tf_core_kernels library
########################################################

if(tensorflow_BUILD_ALL_KERNELS)
  file(GLOB_RECURSE tf_core_kernels_srcs
     "${tensorflow_source_dir}/tensorflow/core/kernels/*.h"
     "${tensorflow_source_dir}/tensorflow/core/kernels/*.cc"
  )
else(tensorflow_BUILD_ALL_KERNELS)
  # Build a minimal subset of kernels to be able to run a test program.
  set(tf_core_kernels_srcs
     "${tensorflow_source_dir}/tensorflow/core/kernels/bounds_check.h"
     "${tensorflow_source_dir}/tensorflow/core/kernels/constant_op.h"
     "${tensorflow_source_dir}/tensorflow/core/kernels/constant_op.cc"
     "${tensorflow_source_dir}/tensorflow/core/kernels/fill_functor.h"
     "${tensorflow_source_dir}/tensorflow/core/kernels/fill_functor.cc"
     "${tensorflow_source_dir}/tensorflow/core/kernels/matmul_op.h"
     "${tensorflow_source_dir}/tensorflow/core/kernels/matmul_op.cc"
     "${tensorflow_source_dir}/tensorflow/core/kernels/no_op.h"
     "${tensorflow_source_dir}/tensorflow/core/kernels/no_op.cc"
     "${tensorflow_source_dir}/tensorflow/core/kernels/sendrecv_ops.h"
     "${tensorflow_source_dir}/tensorflow/core/kernels/sendrecv_ops.cc"
  )
endif(tensorflow_BUILD_ALL_KERNELS)

if(tensorflow_BUILD_CONTRIB_KERNELS)
  set(tf_contrib_kernels_srcs
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/clustering_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/kernels/wals_solver_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/clustering_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/factorization_ops.cc"
      #"${tensorflow_source_dir}/tensorflow/contrib/ffmpeg/decode_audio_op.cc"
      #"${tensorflow_source_dir}/tensorflow/contrib/ffmpeg/encode_audio_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/layers/kernels/bucketization_kernel.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/layers/kernels/sparse_feature_cross_kernel.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/layers/ops/bucketization_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/layers/ops/sparse_feature_cross_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/blas_gemm.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/gru_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/lstm_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/gru_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/lstm_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/tensor_forest_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/best_splits_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/count_extremely_random_stats_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/finished_nodes_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/grow_tree_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/reinterpret_string_to_float_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/sample_inputs_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/scatter_add_ndim_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/topn_ops.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/tree_predictions_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/tree_utils.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/core/ops/update_fertile_slots_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/hard_routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/k_feature_gradient_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/k_feature_routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/routing_gradient_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/stochastic_hard_routing_function_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/stochastic_hard_routing_gradient_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/unpack_path_op.cc"
      "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/utils.cc"
    )
  list(APPEND tf_core_kernels_srcs ${tf_contrib_kernels_srcs})
endif(tensorflow_BUILD_CONTRIB_KERNELS)

if(NOT tensorflow_ENABLE_SSL_SUPPORT)
  # Cloud libraries require boringssl.
  file(GLOB tf_core_kernels_cloud_srcs
      "${tensorflow_source_dir}/tensorflow/core/kernels/cloud/*.h"
      "${tensorflow_source_dir}/tensorflow/core/kernels/cloud/*.cc"
  )
list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_cloud_srcs})
endif()

file(GLOB_RECURSE tf_core_kernels_exclude_srcs
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test*.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*test*.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*testutil.h"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*testutil.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*main.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/*.cu.cc"
   "${tensorflow_source_dir}/tensorflow/core/kernels/debug_ops.h"  # stream_executor dependency
   "${tensorflow_source_dir}/tensorflow/core/kernels/debug_ops.cc"  # stream_executor dependency
)
list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_exclude_srcs})

if(WIN32)
  file(GLOB_RECURSE tf_core_kernels_windows_exclude_srcs
      # not working on windows yet
      "${tensorflow_source_dir}/tensorflow/core/kernels/meta_support.*"
      "${tensorflow_source_dir}/tensorflow/core/kernels/*quantiz*.h"
      "${tensorflow_source_dir}/tensorflow/core/kernels/*quantiz*.cc"
  )
  list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_windows_exclude_srcs})
endif(WIN32)

file(GLOB_RECURSE tf_core_gpu_kernels_srcs
   "${tensorflow_source_dir}/tensorflow/core/kernels/*.cu.cc"
   "${tensorflow_source_dir}/tensorflow/contrib/rnn/kernels/*.cu.cc"
)

if(WIN32 AND tensorflow_ENABLE_GPU)
  file(GLOB_RECURSE tf_core_kernels_cpu_only_srcs
      # GPU implementation not working on Windows yet.
      "${tensorflow_source_dir}/tensorflow/core/kernels/matrix_diag_op.cc"
      "${tensorflow_source_dir}/tensorflow/core/kernels/one_hot_op.cc")
  list(REMOVE_ITEM tf_core_kernels_srcs ${tf_core_kernels_cpu_only_srcs})
  add_library(tf_core_kernels_cpu_only OBJECT ${tf_core_kernels_cpu_only_srcs})
  add_dependencies(tf_core_kernels_cpu_only tf_core_cpu)
  # Undefine GOOGLE_CUDA to avoid registering unsupported GPU kernel symbols.
  get_target_property(target_compile_flags tf_core_kernels_cpu_only COMPILE_FLAGS)
  if(target_compile_flags STREQUAL "target_compile_flags-NOTFOUND")
    set(target_compile_flags "/UGOOGLE_CUDA")
  else()
    set(target_compile_flags "${target_compile_flags} /UGOOGLE_CUDA")
  endif()
  set_target_properties(tf_core_kernels_cpu_only PROPERTIES COMPILE_FLAGS ${target_compile_flags})
endif(WIN32 AND tensorflow_ENABLE_GPU)

add_library(tf_core_kernels OBJECT ${tf_core_kernels_srcs})
add_dependencies(tf_core_kernels tf_core_cpu)

if(WIN32)
  target_compile_options(tf_core_kernels PRIVATE /MP)
  if (tensorflow_ENABLE_GPU)
    set_source_files_properties(${tf_core_gpu_kernels_srcs} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(tf_core_gpu_kernels_lib tf_core_gpu_kernels)
    cuda_add_library(${tf_core_gpu_kernels_lib} ${tf_core_gpu_kernels_srcs})
    set_target_properties(${tf_core_gpu_kernels_lib}
                          PROPERTIES DEBUG_POSTFIX ""
                          COMPILE_FLAGS "${TF_REGULAR_CXX_FLAGS}"
    )
    add_dependencies(${tf_core_gpu_kernels_lib} tf_core_cpu)
  endif()
endif()
