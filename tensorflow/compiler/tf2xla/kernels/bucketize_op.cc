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

#include <algorithm>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/arithmetic.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

class BucketizeOp : public XlaOpKernel {
 public:
  explicit BucketizeOp(OpKernelConstruction* context) : XlaOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("boundaries", &boundaries_));
    OP_REQUIRES(context, std::is_sorted(boundaries_.begin(), boundaries_.end()),
                errors::InvalidArgument("Expected sorted boundaries"));
  }

  void Compile(XlaOpKernelContext* context) override {
    xla::XlaBuilder* builder = context->builder();
    const DataType dtype = context->input_type(0);
    xla::XlaOp input = context->Input(0);

    xla::XlaOp boundaries = xla::ConstantR1<float>(builder, boundaries_);
    // TODO(phawkins): the following behavior matches the behavior of the core
    // Bucketize kernel. However, comparing an int32 or int64 against float may
    // lead to inaccurate bucketing due to rounding.
    if (dtype == DT_DOUBLE) {
      input = xla::ConvertElementType(input, xla::F64);
      boundaries = xla::ConvertElementType(boundaries, xla::F64);
    } else {
      input = xla::ConvertElementType(input, xla::F32);
    }
    xla::XlaOp comparison =
        xla::ConvertElementType(xla::Ge(xla::Broadcast(input, {1}), boundaries,
                                        /*broadcast_dimensions=*/{0}),
                                xla::S32);
    xla::XlaOp buckets = xla::Reduce(
        comparison, /*init_value=*/xla::ConstantR0<int32>(builder, 0),
        /*computation=*/xla::CreateScalarAddComputation(xla::S32, builder),
        /*dimensions_to_reduce=*/{0});
    context->SetOutput(0, buckets);
  }

 private:
  std::vector<float> boundaries_;
};

REGISTER_XLA_OP(Name("Bucketize"), BucketizeOp);

}  // namespace
}  // namespace tensorflow
