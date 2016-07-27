/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

template <typename T>
class SummaryTensorOp : public OpKernel {
 public:
  explicit SummaryTensorOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& tensor = c->input(0);

    Summary s;
    Summary::Value* v = s.add_value();
    v->set_node_name(c->op_kernel().name());

    if (tensor.dtype() == DT_STRING) {
      // tensor_util.makeNdarray doesn't work for strings in tensor_content
      tensor.AsProtoField(v->mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(v->mutable_tensor());
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(s.SerializeToString(&summary_tensor->scalar<string>()()));
  }
};

#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("TensorSummary").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryTensorOp<T>);

TF_CALL_ALL_TYPES(REGISTER)

#undef REGISTER

}  // namespace tensorflow
