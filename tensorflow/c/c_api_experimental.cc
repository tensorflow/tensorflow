/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/c/c_api_experimental.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/compiler/jit/legacy_flags/mark_for_compilation_pass_flags.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/config.pb.h"

using tensorflow::Status;

void TF_EnableXLACompilation(TF_SessionOptions* options, unsigned char enable) {
  tensorflow::ConfigProto& config = options->options.config;
  auto* optimizer_options =
      config.mutable_graph_options()->mutable_optimizer_options();
  if (enable) {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::ON_1);

    // These XLA flags are needed to trigger XLA properly from C (more generally
    // non-Python) clients. If this API is called again with `enable` set to
    // false, it is safe to keep these flag values as is.
    tensorflow::legacy_flags::MarkForCompilationPassFlags* flags =
        tensorflow::legacy_flags::GetMarkForCompilationPassFlags();
    flags->tf_xla_cpu_global_jit = true;
    flags->tf_xla_min_cluster_size = 1;
  } else {
    optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  }
}

void TF_InitializeTPU(TF_Session* session, TF_Status* status) {
  VLOG(1) << "Initializing TPU";
  TF_Operation* config_op =
      TF_GraphOperationByName(session->graph, "ConfigureDistributedTPU");
  if (config_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find node ConfigureDistributedTPU in the TF graph.");
    return;
  }

  TF_Output config_node{config_op, 0};

  TF_Tensor* dummy_output;
  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ &config_node, /*output_values*/ &dummy_output,
                /*noutputs*/ 1,
                /*targets*/ nullptr, /*ntargets*/ 0,
                /*run_metadata*/ nullptr, status);
  if (status->status.ok()) {
    TF_DeleteTensor(dummy_output);
  }
}

void TF_ShutdownTPU(TF_Session* session, TF_Status* status) {
  {
    tensorflow::mutex_lock c(session->graph->mu);
    VLOG(1) << "Shutting down TPU, with input graph: "
            << session->graph->graph.ToGraphDefDebug().DebugString();
  }

  TF_Operation* shutdown_op =
      TF_GraphOperationByName(session->graph, "ShutdownDistributedTPU");
  if (shutdown_op == nullptr) {
    status->status = tensorflow::errors::Internal(
        "Unable to find node ShutdownDistributedTPU in the TF graph.");
    return;
  }

  TF_SessionRun(session, /*run_options*/ nullptr,
                // input related parameters
                /*inputs*/ nullptr, /*input_values*/ nullptr, /*ninputs*/ 0,
                // output related parameters
                /*outputs*/ nullptr, /*output_values*/ nullptr,
                /*noutputs*/ 0,
                /*targets*/ &shutdown_op, /*ntargets*/ 1,
                /*run_metadata*/ nullptr, status);
}

TF_CAPI_EXPORT extern const char* TF_GraphDebugString(TF_Graph* graph,
                                                      size_t* len) {
  tensorflow::mutex_lock c(graph->mu);
  const auto& debug_str = graph->graph.ToGraphDefDebug().DebugString();
  *len = debug_str.size();
  char* ret = static_cast<char*>(malloc(*len + 1));
  memcpy(ret, debug_str.c_str(), *len + 1);
  return ret;
}
