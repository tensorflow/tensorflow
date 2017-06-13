/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

REGISTER_OP("KernelLabel")
    .Output("result: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GraphDefVersion")
    .Output("version: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("RequiresOlderGraphVersion")
    .Output("version: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->graph_def_version() != TF_GRAPH_DEF_VERSION - 1) {
        return errors::InvalidArgument("Wrong graph version for shape");
      }
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("Old")
    .SetShapeFn(shape_inference::UnknownShape)
    .Deprecated(8, "For reasons");

REGISTER_RESOURCE_HANDLE_OP(StubResource);

REGISTER_OP("ResourceInitializedOp")
    .Input("resource: resource")
    .Output("initialized: bool")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ResourceCreateOp")
    .Input("resource: resource")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("ResourceUsingOp")
    .Input("resource: resource")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("TestStringOutput")
    .Input("input: float")
    .Output("output1: float")
    .Output("output2: string")
    .SetShapeFn(shape_inference::UnknownShape);

namespace {
enum KernelLabel { DEFAULT_LABEL, OVERLOAD_1_LABEL, OVERLOAD_2_LABEL };
}  // namespace

template <KernelLabel KL>
class KernelLabelOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("result", TensorShape({}), &output));
    switch (KL) {
      case DEFAULT_LABEL:
        output->scalar<string>()() = "My label is: default";
        break;
      case OVERLOAD_1_LABEL:
        output->scalar<string>()() = "My label is: overload_1";
        break;
      case OVERLOAD_2_LABEL:
        output->scalar<string>()() = "My label is: overload_2";
        break;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KernelLabel").Device(DEVICE_CPU),
                        KernelLabelOp<DEFAULT_LABEL>);
REGISTER_KERNEL_BUILDER(Name("KernelLabel")
                            .Device(DEVICE_CPU)
                            .Label("overload_1"),
                        KernelLabelOp<OVERLOAD_1_LABEL>);
REGISTER_KERNEL_BUILDER(Name("KernelLabel")
                            .Device(DEVICE_CPU)
                            .Label("overload_2"),
                        KernelLabelOp<OVERLOAD_2_LABEL>);

class GraphDefVersionOp : public OpKernel {
 public:
  explicit GraphDefVersionOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {}

  void Compute(OpKernelContext* ctx) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->scalar<int>()() = graph_def_version_;
  }

 private:
  const int graph_def_version_;
};

REGISTER_KERNEL_BUILDER(Name("GraphDefVersion").Device(DEVICE_CPU),
                        GraphDefVersionOp);

class OldOp : public OpKernel {
 public:
  explicit OldOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {}
};

REGISTER_KERNEL_BUILDER(Name("Old").Device(DEVICE_CPU), OldOp);

// Stubbed-out resource to test resource handle ops.
class StubResource : public ResourceBase {
 public:
  string DebugString() override { return ""; }
};

REGISTER_RESOURCE_HANDLE_KERNEL(StubResource);

REGISTER_KERNEL_BUILDER(Name("ResourceInitializedOp").Device(DEVICE_CPU),
                        IsResourceInitialized<StubResource>);

class ResourceCreateOp : public OpKernel {
 public:
  explicit ResourceCreateOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    OP_REQUIRES_OK(c,
                   CreateResource(c, HandleFromInput(c, 0), new StubResource));
  }
};

REGISTER_KERNEL_BUILDER(Name("ResourceCreateOp").Device(DEVICE_CPU),
                        ResourceCreateOp);

// Uses a ResourceHandle to check its validity.
class ResourceUsingOp : public OpKernel {
 public:
  explicit ResourceUsingOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    StubResource* unused;
    OP_REQUIRES_OK(ctx, LookupResource<StubResource>(
                            ctx, HandleFromInput(ctx, 0), &unused));
  }
};

REGISTER_KERNEL_BUILDER(Name("ResourceUsingOp").Device(DEVICE_CPU),
                        ResourceUsingOp);

}  // end namespace tensorflow
