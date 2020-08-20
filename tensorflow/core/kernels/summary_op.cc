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

class SummaryMergeOp : public OpKernel {
 public:
  explicit SummaryMergeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    Summary s;
    std::unordered_set<string> tags;
    for (int input_num = 0; input_num < c->num_inputs(); input_num++) {
      const Tensor& in = c->input(input_num);
      auto in_vec = in.flat<tstring>();
      for (int i = 0; i < in_vec.dimension(0); i++) {
        const string& s_in = in_vec(i);
        Summary summary_in;
        if (!ParseProtoUnlimited(&summary_in, s_in)) {
          c->SetStatus(errors::InvalidArgument(
              "Could not parse one of the summary inputs"));
          return;
        }

        for (int v = 0; v < summary_in.value_size(); v++) {
          const string& tag = summary_in.value(v).tag();
          // The tag is unused by the TensorSummary op, so no need to check
          // for duplicates.
          if ((!tag.empty()) && !tags.insert(tag).second) {
            c->SetStatus(errors::InvalidArgument(strings::StrCat(
                "Duplicate tag ", tag, " found in summary inputs")));
            return;
          }
          *s.add_value() = summary_in.value(v);
        }
      }
    }

    Tensor* summary_tensor = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({}), &summary_tensor));
    CHECK(SerializeToTString(s, &summary_tensor->scalar<tstring>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("MergeSummary").Device(DEVICE_CPU),
                        SummaryMergeOp);

}  // namespace tensorflow
