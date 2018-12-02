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

#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {

// XLA_* devices also register a "real" NoOp operator so we suppress the
// dummy operator using CompilationOnly().
REGISTER_XLA_OP(Name("NoOp").CompilationOnly(), NoOp);

// We register ControlTrigger as a no-op. This is correct since nodes seen
// by the XLA compiler are never dead.
REGISTER_XLA_OP(Name("ControlTrigger").CompilationOnly(), NoOp);

}  // namespace tensorflow
