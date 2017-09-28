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
#include "tensorflow/core/common_runtime/function_testlib.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace test {
namespace function {

typedef FunctionDefHelper FDH;

class FindDeviceOpKernel : public OpKernel {
 public:
  explicit FindDeviceOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* device_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("device_name", TensorShape{},
                                             &device_tensor));
    device_tensor->scalar<string>()() =
        ctx->function_library()->device()->name();
  }
};

REGISTER_KERNEL_BUILDER(Name("FindDeviceOp").Device(tensorflow::DEVICE_CPU),
                        FindDeviceOpKernel);
REGISTER_OP("FindDeviceOp").Output("device_name: string");

FunctionDef FindDevice() {
  return FDH::Define(
      // Name
      "FindDevice",
      // Args
      {},
      // Return values
      {"device_name: string"},
      // Attr def
      {},
      // Nodes
      {{{"device_name"}, "FindDeviceOp", {}, {}}});
}

// TODO(phawkins): replace with C++ API for calling functions, when that exists.
Output Call(Scope* scope, const string& op_name, const string& fn_name,
            gtl::ArraySlice<Input> inputs) {
  NodeDef def;
  NodeDefBuilder builder(op_name, fn_name, scope->graph()->op_registry());
  for (const Input& input : inputs) {
    builder.Input(input.node()->name(), input.index(),
                  input.node()->output_type(input.index()));
  }
  TF_CHECK_OK(builder.Finalize(&def));
  Status status;
  Node* n = scope->graph()->AddNode(def, &status);
  TF_CHECK_OK(status);
  TF_CHECK_OK(scope->DoShapeInference(n));
  for (int i = 0; i < inputs.size(); ++i) {
    scope->graph()->AddEdge(inputs[i].node(), inputs[i].index(), n, i);
  }
  return Output(n);
}

}  // namespace function
}  // namespace test
}  // namespace tensorflow
