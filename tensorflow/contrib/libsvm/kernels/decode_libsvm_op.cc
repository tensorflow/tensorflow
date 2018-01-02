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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

template <typename T, typename Tlabel>
class DecodeLibsvmOp : public OpKernel {
 public:
  explicit DecodeLibsvmOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_features", &num_features_));
    OP_REQUIRES(ctx, (num_features_ >= 1),
                errors::InvalidArgument("Invalid number of features \"",
                                        num_features_, "\""));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* label_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor->shape(), &label_tensor));
    auto label = label_tensor->flat<Tlabel>();

    std::vector<T> out_values;
    std::vector<std::pair<int64, int64>> out_indices;
    for (int i = 0; i < input_flat.size(); ++i) {
      std::vector<string> entries =
          str_util::Split(input_flat(i), " ", str_util::SkipEmpty());
      OP_REQUIRES(ctx, !entries.empty(),
                  errors::InvalidArgument("No entries found for input[", i,
                                          "]: \"", input_flat(i), "\""));
      Tlabel label_value;
      OP_REQUIRES(
          ctx, strings::SafeStringToNumeric<Tlabel>(entries[0], &label_value),
          errors::InvalidArgument("Label format incorrect: ", entries[0]));
      label(i) = label_value;
      for (int j = 1; j < entries.size(); j++) {
        std::vector<string> pair = str_util::Split(entries[j], ":");
        OP_REQUIRES(
            ctx, (pair.size() == 2),
            errors::InvalidArgument("Invalid feature \"", entries[j], "\""));
        int64 feature_index;
        OP_REQUIRES(
            ctx, strings::safe_strto64(pair[0].c_str(), &feature_index),
            errors::InvalidArgument("Feature format incorrect: ", entries[j]));
        OP_REQUIRES(ctx, (feature_index >= 0),
                    errors::InvalidArgument(
                        "Feature index should be >= 0, got ", feature_index));
        T feature_value;
        OP_REQUIRES(
            ctx, strings::SafeStringToNumeric<T>(pair[1], &feature_value),
            errors::InvalidArgument("Feature format incorrect: ", entries[j]));
        out_values.emplace_back(feature_value);
        out_indices.emplace_back(std::pair<int64, int64>(i, feature_index));
      }
    }

    Tensor* indices_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1,
                            TensorShape({static_cast<int64>(out_indices.size()),
                                         input_tensor->shape().dims() + 1}),
                            &indices_tensor));
    auto indices = indices_tensor->matrix<int64>();
    // Translate flat index to shaped index like np.unravel_index
    // Calculate factors for each dimension
    std::vector<int64> factors(input_tensor->shape().dims());
    factors[input_tensor->shape().dims() - 1] = 1;
    for (int j = input_tensor->shape().dims() - 2; j >= 0; j--) {
      factors[j] = factors[j + 1] * input_tensor->shape().dim_size(j + 1);
    }
    for (int i = 0; i < out_indices.size(); i++) {
      indices(i, 0) = out_indices[i].first;
      int64 value = out_indices[i].first;
      for (int j = 0; j < input_tensor->shape().dims(); j++) {
        indices(i, j) = value / factors[j];
        value = value % factors[j];
      }
      indices(i, input_tensor->shape().dims()) = out_indices[i].second;
    }

    Tensor* values_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(
                       2, TensorShape({static_cast<int64>(out_values.size())}),
                       &values_tensor));
    auto values = values_tensor->vec<T>();
    std::copy_n(out_values.begin(), out_values.size(), &values(0));

    Tensor* shape_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            3, TensorShape({input_tensor->shape().dims() + 1}),
                            &shape_tensor));
    auto shape = shape_tensor->flat<int64>();
    for (int i = 0; i < input_tensor->shape().dims(); i++) {
      shape(i) = input_tensor->shape().dim_size(i);
    }
    shape(input_tensor->shape().dims()) = num_features_;
  }

 private:
  int64 num_features_;
};

#define REGISTER_KERNEL(type)                                         \
  REGISTER_KERNEL_BUILDER(Name("DecodeLibsvm")                        \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("dtype")          \
                              .TypeConstraint<int32>("label_dtype"),  \
                          DecodeLibsvmOp<type, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("DecodeLibsvm")                        \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("dtype")          \
                              .TypeConstraint<int64>("label_dtype"),  \
                          DecodeLibsvmOp<type, int64>);               \
  REGISTER_KERNEL_BUILDER(Name("DecodeLibsvm")                        \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("dtype")          \
                              .TypeConstraint<float>("label_dtype"),  \
                          DecodeLibsvmOp<type, float>);               \
  REGISTER_KERNEL_BUILDER(Name("DecodeLibsvm")                        \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<type>("dtype")          \
                              .TypeConstraint<double>("label_dtype"), \
                          DecodeLibsvmOp<type, double>);

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
#undef REGISTER_KERNEL

}  // namespace tensorflow
