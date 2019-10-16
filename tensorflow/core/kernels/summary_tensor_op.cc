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
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

template <typename T>
class SummaryTensorOpV2 : public OpKernel {
 public:
  explicit SummaryTensorOpV2(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& tag = c->input(0);
    OP_REQUIRES(c, IsLegacyScalar(tag.shape()),
                errors::InvalidArgument("tag must be scalar"));
    const Tensor& tensor = c->input(1);
    const Tensor& serialized_summary_metadata_tensor = c->input(2);

    Summary s;
    Summary::Value* v = s.add_value();
    v->set_tag(string(tag.scalar<tstring>()()));  // NOLINT

    if (tensor.dtype() == DT_STRING) {
      // tensor_util.makeNdarray doesn't work for strings in tensor_content
      tensor.AsProtoField(v->mutable_tensor());
    } else {
      tensor.AsProtoTensorContent(v->mutable_tensor());
    }

    v->mutable_metadata()->ParseFromString(
        serialized_summary_metadata_tensor.scalar<tstring>()());

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};

#define REGISTER(T)                                                      \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("TensorSummaryV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryTensorOpV2<T>);

TF_CALL_ALL_TYPES(REGISTER)

#undef REGISTER

// NOTE(chizeng): We are phasing out the use of SummaryTensorOp in favor of
// SummaryTensorOpV2. This is because SummaryTensorOpV2 allows the callers to
// pass a tag (more consistent with other summaries) as well as serialized
// summary metadata used by plugins (which lets TensorBoard determine which
// events are relevant to which plugins).
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
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};

#define REGISTER(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("TensorSummary").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryTensorOp<T>);

TF_CALL_ALL_TYPES(REGISTER)

#undef REGISTER

}  // namespace tensorflow
