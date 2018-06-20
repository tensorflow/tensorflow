/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace tensorflow {

bool GpuOpFilter(KernelDef* kdef) {
  // TODO(b/31361304): The GPU backend does not parallelize PRNG ops, leading to
  // slow code.
  if (kdef->op() == "RandomStandardNormal" || kdef->op() == "RandomUniform" ||
      kdef->op() == "RandomUniformInt" || kdef->op() == "TruncatedNormal") {
    return false;
  }
  if (kdef->op() == "Const") {
    AddDtypeToKernalDefConstraint("dtype", DT_STRING, kdef);
  }
  if (kdef->op() == "Assert") {
    AddDtypeToKernalDefConstraint("T", DT_STRING, kdef);
  }
  return true;
}

REGISTER_XLA_BACKEND(DEVICE_GPU_XLA_JIT, kGpuAllTypes, GpuOpFilter);

}  // namespace tensorflow
