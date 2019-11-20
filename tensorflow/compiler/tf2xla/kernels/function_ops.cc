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

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {
namespace {

const char* const kGradientOp = "SymbolicGradient";

// Implementations of _ListToArray and _ArrayToList for functions.
class PassOn : public XlaOpKernel {
 public:
  explicit PassOn(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES(ctx, ctx->num_inputs() == ctx->num_outputs(),
                errors::Internal("#inputs != #outputs : ", ctx->num_inputs(),
                                 " vs. ", ctx->num_outputs()));
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(
          ctx, input_type(i) == output_type(i),
          errors::Internal("Input and output types for position ", i,
                           " do not match: ", DataTypeString(input_type(i)),
                           " vs. ", DataTypeString(output_type(i))));
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      ctx->SetOutput(i, ctx->Input(i));
    }
  }
};

REGISTER_XLA_OP(Name("_ListToArray"), PassOn);
REGISTER_XLA_OP(Name("_ArrayToList"), PassOn);

class AlwaysFailOp : public OpKernel {
 public:
  explicit AlwaysFailOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  ~AlwaysFailOp() override {}

  void Compute(OpKernelContext* ctx) override {
    ctx->CtxFailure(errors::FailedPrecondition(
        "Unexpected attempt to compile ", name(), " which is a ", type_string(),
        ".  These nodes should always be handled by the graph compiler"));
  }
};

// These operations are handled specially in the TF/XLA bridge so their
// OpKernel's should never be called.  We still register a dummy kernel so that
// they show up as "supported" when we are deciding whether a graph containing
// them is compilable with XLA.

REGISTER_XLA_OP(Name(kGradientOp), AlwaysFailOp);
REGISTER_XLA_OP(Name("PartitionedCall")
                    .AllowResourceTypes()
                    .AllowVariantTypes()
                    .AllowStringType(),
                AlwaysFailOp);
REGISTER_XLA_OP(Name("StatefulPartitionedCall")
                    .AllowResourceTypes()
                    .AllowVariantTypes()
                    .AllowStringType(),
                AlwaysFailOp);

}  // namespace
}  // namespace tensorflow
