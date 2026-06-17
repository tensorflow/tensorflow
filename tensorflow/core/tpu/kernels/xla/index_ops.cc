/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/index_ops.h"

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

// This registration is needed here because the ArgMax Op is defined in
// third_party where DEVICE_TPU_XLA_JIT is not visible. Most Ops don't need a
// specific TPU whitelist, but ArgMax does because it has a separate CustomCall
// implementation on CPU.
REGISTER_XLA_OP(Name("ArgMax")
                    .Device(DEVICE_TPU_XLA_JIT)
                    .CompileTimeConstantInput("dimension"),
                XlaArgMaxOp);

}  // namespace
}  // namespace tensorflow
