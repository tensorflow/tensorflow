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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace {

class IsFiniteOp : public XlaOpKernel {
 public:
  explicit IsFiniteOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::ComputationDataHandle input = ctx->Input(0);
    ctx->SetOutput(0, ctx->builder()->IsFinite(input));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IsFiniteOp);
};

REGISTER_XLA_OP(Name("IsFinite"), IsFiniteOp);

}  // anonymous namespace
}  // namespace tensorflow
