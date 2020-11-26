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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {
namespace {

class XlaReplicaIdOp : public XlaOpKernel {
 public:
  explicit XlaReplicaIdOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaReplicaIdOp);
};

void XlaReplicaIdOp::Compile(XlaOpKernelContext* ctx) {
  ctx->SetOutput(
      0, xla::ConvertElementType(xla::ReplicaId(ctx->builder()), xla::S32));
}

REGISTER_XLA_OP(Name("XlaReplicaId"), XlaReplicaIdOp);

}  // namespace
}  // namespace tensorflow
