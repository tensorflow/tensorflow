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

// See docs in ../ops/string_ops.cc.

#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class StringJoinOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit StringJoinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("separator", &separator_));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList input_list;
    OP_REQUIRES_OK(context, context->input_list("inputs", &input_list));
    TensorShape input_shape;
    std::vector<bool> is_scalar;
    std::vector<TTypes<tstring>::ConstFlat> inputs;

    for (const auto& input : input_list) {
      inputs.push_back(input.flat<tstring>());
      is_scalar.push_back(TensorShapeUtils::IsScalar(input.shape()));
      if (!TensorShapeUtils::IsScalar(input.shape())) {
        if (TensorShapeUtils::IsScalar(input_shape)) {
          input_shape = input.shape();
        } else {
          OP_REQUIRES(
              context, input_shape == input.shape(),
              errors::InvalidArgument(
                  "Input shapes do not match: ", input_shape.DebugString(),
                  " vs. ", input.shape().DebugString()));
        }
      }
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("output", input_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<tstring>();

    std::vector<absl::string_view> strings(input_list.size());
    for (size_t i = 0; i < input_shape.num_elements(); ++i) {
      for (int j = 0; j < input_list.size(); ++j) {
        strings[j] = (is_scalar[j]) ? inputs[j](0) : inputs[j](i);
      }
      output_flat(i) = absl::StrJoin(strings, separator_);
    }
  }

 private:
  string separator_;
};

REGISTER_KERNEL_BUILDER(Name("StringJoin").Device(DEVICE_CPU), StringJoinOp);

}  // namespace tensorflow
