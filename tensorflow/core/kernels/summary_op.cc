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
class SummaryScalarOp : public OpKernel {
 public:
  explicit SummaryScalarOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& tags = c->input(0);
    const Tensor& values = c->input(1);

    OP_REQUIRES(
        c, tags.IsSameSize(values) ||
               (IsLegacyScalar(tags.shape()) && IsLegacyScalar(values.shape())),
        errors::InvalidArgument("tags and values not the same shape: ",
                                tags.shape().DebugString(), " != ",
                                values.shape().DebugString(), SingleTag(tags)));
    auto Ttags = tags.flat<string>();
    auto Tvalues = values.flat<T>();
    Summary s;
    for (int i = 0; i < Ttags.size(); i++) {
      Summary::Value* v = s.add_value();
      v->set_tag(Ttags(i));
      v->set_simple_value(float(Tvalues(i)));
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(s.SerializeToString(&summary_tensor->scalar<string>()()));
  }

  // If there's only one tag, include it in the error message
  static string SingleTag(const Tensor& tags) {
    if (tags.NumElements() == 1) {
      return strings::StrCat(" (tag '", tags.flat<string>()(0), "')");
    } else {
      return "";
    }
  }
};

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
    OP_REQUIRES(c, IsLegacyScalar(tags.shape()),
                errors::InvalidArgument("tags must be scalar"));
    // Build histogram of values in "values" tensor
    histogram::Histogram histo;
    for (int64 i = 0; i < flat.size(); i++) {
      T v = flat(i);
      if (Eigen::numext::isnan(v)) {
        c->SetStatus(
            errors::InvalidArgument("Nan in summary histogram for: ", name()));
        break;
      } else if (Eigen::numext::isinf(v)) {
        c->SetStatus(errors::InvalidArgument(
            "Infinity in summary histogram for: ", name()));
        break;
      }
      histo.Add(static_cast<double>(v));
    }

    Summary s;
    Summary::Value* v = s.add_value();
    v->set_tag(tags.scalar<string>()());
    histo.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */);

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(s.SerializeToString(&summary_tensor->scalar<string>()()));
  }
};

#define REGISTER(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ScalarSummary").Device(DEVICE_CPU).TypeConstraint<T>("T"),    \
      SummaryScalarOp<T>);                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("HistogramSummary").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SummaryHistoOp<T>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER)
#undef REGISTER

struct HistogramResource : public ResourceBase {
  histogram::ThreadSafeHistogram histogram;

  string DebugString() override { return "A histogram summary. Stats ..."; }
};

class SummaryMergeOp : public OpKernel {
 public:
  explicit SummaryMergeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    Summary s;
    std::unordered_set<string> tags;
    for (int input_num = 0; input_num < c->num_inputs(); input_num++) {
      const Tensor& in = c->input(input_num);
      auto in_vec = in.flat<string>();
      for (int i = 0; i < in_vec.dimension(0); i++) {
        const string& s_in = in_vec(i);
        Summary summary_in;
        if (!ParseProtoUnlimited(&summary_in, s_in)) {
          c->SetStatus(errors::InvalidArgument(
              "Could not parse one of the summary inputs"));
          return;
        }

        for (int v = 0; v < summary_in.value_size(); v++) {
          if (!tags.insert(summary_in.value(v).tag()).second) {
            c->SetStatus(errors::InvalidArgument(
                strings::StrCat("Duplicate tag ", summary_in.value(v).tag(),
                                " found in summary inputs")));
            return;
          }
          *s.add_value() = summary_in.value(v);
        }
      }
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(s.SerializeToString(&summary_tensor->scalar<string>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("MergeSummary").Device(DEVICE_CPU),
                        SummaryMergeOp);

}  // namespace tensorflow
