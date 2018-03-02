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
#include "tensorflow/core/protobuf/config.pb.h"

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
