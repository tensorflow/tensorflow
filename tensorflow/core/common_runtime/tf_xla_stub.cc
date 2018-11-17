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

#include <cstdlib>

#include "tensorflow/core/common_runtime/tf_xla_stub.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

bool is_xla_gpu_jit_registered = false;
bool is_xla_cpu_jit_registered = false;

struct XlaEnvVars {
  bool xla_flags_env_var_present;
  bool tf_xla_flags_env_var_present;
};

XlaEnvVars ComputeEnvVarHasXlaFlags() {
  XlaEnvVars env_vars;
  env_vars.xla_flags_env_var_present = getenv("XLA_FLAGS") != nullptr;
  env_vars.tf_xla_flags_env_var_present = getenv("TF_XLA_FLAGS") != nullptr;
  return env_vars;
}

}  // namespace

XlaGpuJitIsLinkedIn::XlaGpuJitIsLinkedIn() { is_xla_gpu_jit_registered = true; }
XlaCpuJitIsLinkedIn::XlaCpuJitIsLinkedIn() { is_xla_cpu_jit_registered = true; }

Status CheckXlaJitOptimizerOptions(const SessionOptions* session_options) {
  static XlaEnvVars env_vars = ComputeEnvVarHasXlaFlags();

  if (is_xla_cpu_jit_registered || is_xla_gpu_jit_registered) {
    return Status::OK();
  }

  if (env_vars.xla_flags_env_var_present) {
    return errors::InvalidArgument(
        "The XLA JIT is not linked in but the \"XLA_FLAGS\" environment "
        "variable is set.  Please either link in XLA or remove \"XLA_FLAGS\" "
        "from the environment.");
  }

  if (env_vars.tf_xla_flags_env_var_present) {
    return errors::InvalidArgument(
        "The XLA JIT is not linked in but the \"TF_XLA_FLAGS\" environment "
        "variable is set.  Please either link in XLA or remove "
        "\"TF_XLA_FLAGS\" from the environment.");
  }

  if (session_options) {
    OptimizerOptions::GlobalJitLevel jit_level =
        session_options->config.graph_options()
            .optimizer_options()
            .global_jit_level();

    if (jit_level == OptimizerOptions::ON_1 ||
        jit_level == OptimizerOptions::ON_2) {
      return errors::InvalidArgument(
          "The XLA JIT is enabled in the session options but XLA is not linked "
          "in.  Plesae either link in XLA or disable the JIT in the session "
          "options.");
    }
  }

  return Status::OK();
}
}  // namespace tensorflow
