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

// Operators that deal with SummaryProtos (encoded as DT_STRING tensors) as
// inputs or outputs in various ways.

// See docs in ../ops/summary_ops.cc.

#include <unordered_set>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

template <typename T>
class SummaryHistoOp : public OpKernel {
 public:
  // SummaryHistoOp could be extended to take a list of custom bucket
  // boundaries as an option.
  explicit SummaryHistoOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& tags = c->input(0);
    const Tensor& values = c->input(1);
    const auto flat = values.flat<T>();
    OP_REQUIRES(c, TensorShapeUtils::IsScalar(tags.shape()),
                errors::InvalidArgument("tags must be scalar"));
    // Build histogram of values in "values" tensor
    histogram::Histogram histo;
    for (int64 i = 0; i < flat.size(); i++) {
      const double double_val = static_cast<double>(flat(i));
      if (Eigen::numext::isnan(double_val)) {
        c->SetStatus(
            errors::InvalidArgument("Nan in summary histogram for: ", name()));
        break;
      } else if (Eigen::numext::isinf(double_val)) {
        c->SetStatus(errors::InvalidArgument(
            "Infinity in summary histogram for: ", name()));
        break;
      }
      histo.Add(double_val);
    }

    Summary s;
    Summary::Value* v = s.add_value();
    const tstring& tags0 = tags.scalar<tstring>()();
    v->set_tag(tags0.data(), tags0.size());
    histo.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */);

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};

#define REGISTER(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("HistogramSummary").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryHistoOp<T>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER)
#undef REGISTER

struct HistogramResource : public ResourceBase {
  histogram::ThreadSafeHistogram histogram;

  string DebugString() const override {
    return "A histogram summary. Stats ...";
  }
};

}  // namespace tensorflow
