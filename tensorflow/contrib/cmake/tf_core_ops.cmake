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
set(tf_op_lib_names
    "audio_ops"
    "array_ops"
    "batch_ops"
    "bitwise_ops"
    "boosted_trees_ops"
    "candidate_sampling_ops"
    "checkpoint_ops"
    "control_flow_ops"
    "ctc_ops"
    "cudnn_rnn_ops"
    "data_flow_ops"
    "dataset_ops"
    "decode_proto_ops"
    "encode_proto_ops"
    "functional_ops"
    "image_ops"
    "io_ops"
    "linalg_ops"
    "list_ops"
    "lookup_ops"
    "logging_ops"
    "manip_ops"
    "math_ops"
    "nn_ops"
    "no_op"
    "parsing_ops"
    "random_ops"
    "remote_fused_graph_ops"
    "resource_variable_ops"
    "rpc_ops"
    "script_ops"
    "sdca_ops"
    "set_ops"
    "sendrecv_ops"
    "sparse_ops"
    "spectral_ops"
    "state_ops"
    "stateless_random_ops"
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
endforeach()

function(GENERATE_CONTRIB_OP_LIBRARY op_lib_name cc_srcs)
    add_library(tf_contrib_${op_lib_name}_ops OBJECT ${cc_srcs})
    add_dependencies(tf_contrib_${op_lib_name}_ops tf_core_framework)
endfunction()

file(GLOB_RECURSE tensor_forest_hybrid_srcs
     "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/hybrid/core/ops/*.cc"
)

file(GLOB_RECURSE tpu_ops_srcs
     "${tensorflow_source_dir}/tensorflow/contrib/tpu/ops/*.cc"
)

GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_model "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/model_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_split_handler "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/split_handler_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_training "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/training_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_prediction "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/prediction_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_quantiles "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/quantile_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(boosted_trees_stats_accumulator "${tensorflow_source_dir}/tensorflow/contrib/boosted_trees/ops/stats_accumulator_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(coder "${tensorflow_source_dir}/tensorflow/contrib/coder/ops/coder_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(factorization_clustering "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/clustering_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(factorization_factorization "${tensorflow_source_dir}/tensorflow/contrib/factorization/ops/factorization_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(framework_variable "${tensorflow_source_dir}/tensorflow/contrib/framework/ops/variable_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(input_pipeline "${tensorflow_source_dir}/tensorflow/contrib/input_pipeline/ops/input_pipeline_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(image "${tensorflow_source_dir}/tensorflow/contrib/image/ops/image_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(image_distort_image "${tensorflow_source_dir}/tensorflow/contrib/image/ops/distort_image_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(image_sirds "${tensorflow_source_dir}/tensorflow/contrib/image/ops/single_image_random_dot_stereograms_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(layers_sparse_feature_cross "${tensorflow_source_dir}/tensorflow/contrib/layers/ops/sparse_feature_cross_op.cc")
GENERATE_CONTRIB_OP_LIBRARY(memory_stats "${tensorflow_source_dir}/tensorflow/contrib/memory_stats/ops/memory_stats_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(periodic_resample "${tensorflow_source_dir}/tensorflow/contrib/periodic_resample/ops/array_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(nearest_neighbor "${tensorflow_source_dir}/tensorflow/contrib/nearest_neighbor/ops/nearest_neighbor_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(resampler "${tensorflow_source_dir}/tensorflow/contrib/resampler/ops/resampler_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(rnn_gru "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/gru_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(rnn_lstm "${tensorflow_source_dir}/tensorflow/contrib/rnn/ops/lstm_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(seq2seq_beam_search "${tensorflow_source_dir}/tensorflow/contrib/seq2seq/ops/beam_search_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/tensor_forest_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest_hybrid "${tensor_forest_hybrid_srcs}")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest_model "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/model_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tensor_forest_stats "${tensorflow_source_dir}/tensorflow/contrib/tensor_forest/ops/stats_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(text_skip_gram "${tensorflow_source_dir}/tensorflow/contrib/text/ops/skip_gram_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(tpu "${tpu_ops_srcs}")
GENERATE_CONTRIB_OP_LIBRARY(bigquery_reader "${tensorflow_source_dir}/tensorflow/contrib/cloud/ops/bigquery_reader_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(gcs_config "${tensorflow_source_dir}/tensorflow/contrib/cloud/ops/gcs_config_ops.cc")
GENERATE_CONTRIB_OP_LIBRARY(reduce_slice_ops "${tensorflow_source_dir}/tensorflow/contrib/reduce_slice_ops/ops/reduce_slice_ops.cc")

########################################################
# tf_user_ops library
########################################################
file(GLOB_RECURSE tf_user_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/user_ops/*.cc"
)

add_library(tf_user_ops OBJECT ${tf_user_ops_srcs})

add_dependencies(tf_user_ops tf_core_framework)

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

########################################################
# tf_debug_ops library
########################################################

file(GLOB tf_debug_ops_srcs
    "${tensorflow_source_dir}/tensorflow/core/ops/debug_ops.cc"
)

add_library(tf_debug_ops OBJECT ${tf_debug_ops_srcs})

add_dependencies(tf_debug_ops tf_core_framework)
