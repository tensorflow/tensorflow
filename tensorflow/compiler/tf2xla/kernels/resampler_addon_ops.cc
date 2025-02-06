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

#include "tensorflow/compiler/tf2xla/kernels/resampler_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

REGISTER_XLA_OP(Name("Addons>Resampler")
                    .TypeConstraint("T", {DT_HALF, DT_FLOAT, DT_DOUBLE}),
                ResamplerOp);

REGISTER_XLA_OP(Name("Addons>ResamplerGrad")
                    .TypeConstraint("T", {DT_HALF, DT_FLOAT, DT_DOUBLE}),
                ResamplerGradOp);
}  // namespace
}  // namespace tensorflow
