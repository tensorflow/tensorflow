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

#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// This test is to verify the correctness of XLA op registration with specific
// backend overrides.

// A dummy backend-specific OpKernel for CPU.
class DummyCPUOp : public XlaOpKernel {
 public:
  explicit DummyCPUOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, ctx->Input(0));
  }
};

// A dummy generic OpKernel for all backends.
class DummyGenericOp : public XlaOpKernel {
 public:
  explicit DummyGenericOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, ctx->Input(0));
  }
};

REGISTER_OP("DummyDuplicateOp")
    .Attr("T: {float, int32}")
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
A dummy Op.

input: dummy input.
output: dummy output.
)doc");

// Register the DummyCPUOp kernel for CPU with type INT32.
REGISTER_XLA_OP(Name("DummyDuplicateOp")
                    .Device(DEVICE_CPU_XLA_JIT)
                    .TypeConstraint("T", DT_INT32),
                DummyCPUOp);
// Register the DummyGeneric kernel for all registered device (except CPU since
// it is already registered), with type FLOAT.
REGISTER_XLA_OP(Name("DummyDuplicateOp").TypeConstraint("T", DT_FLOAT),
                DummyGenericOp);

// Test the correctness of registered kernels. The kernel registered for CPU
// should have type INT32 while all other kernels should have type FLOAT.
TEST(XlaOpRegistryTest, XlaOpRegistrationWithOverride) {
  XlaOpRegistry::RegisterCompilationKernels();
  auto registered_kernels = GetAllRegisteredKernels().kernel();
  for (const auto& kernels : registered_kernels) {
    if (kernels.op() == "DummyDuplicateOp") {
      EXPECT_EQ(kernels.constraint_size(), 1);
      EXPECT_EQ(kernels.constraint(0).name(), "T");
      if (kernels.device_type() == "XLA_CPU_JIT") {
        EXPECT_EQ(kernels.constraint(0).allowed_values().list().type(0),
                  DT_INT32);
      } else {
        EXPECT_EQ(kernels.constraint(0).allowed_values().list().type(0),
                  DT_FLOAT);
      }
    }
  }
}

// A dummy generic OpKernel for all backends.
class DummyInfeasibleTypeConstraintOp : public XlaOpKernel {
 public:
  explicit DummyInfeasibleTypeConstraintOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    LOG(FATAL) << "unreachable";
  }
};

REGISTER_OP("DummyInfeasibleTypeConstraintOp")
    .Attr("T: {float, string}")
    .Input("input: T")
    .Output("output: T")
    .Doc(R"doc(
A dummy Op.

input: dummy input.
output: dummy output.
)doc");
REGISTER_XLA_OP(
    Name("DummyInfeasibleTypeConstraintOp").TypeConstraint("T", DT_STRING),
    DummyInfeasibleTypeConstraintOp);

TEST(XlaOpRegistryTest, OpWithInfeasibleTypeConstraintIsNotRegistered) {
  XlaOpRegistry::RegisterCompilationKernels();
  auto registered_kernels = GetAllRegisteredKernels().kernel();
  for (const auto& kernels : registered_kernels) {
    // The operator should not be registered.
    EXPECT_NE(kernels.op(), "DummyInfeasibleTypeConstraintOp");
  }
}

}  // namespace
}  // namespace tensorflow
