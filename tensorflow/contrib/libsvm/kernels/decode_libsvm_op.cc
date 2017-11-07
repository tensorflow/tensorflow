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

template <typename T>
class DecodeLibsvmOp : public OpKernel {
 public:
  explicit DecodeLibsvmOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_features", &num_features_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* label_tensor;
    Tensor* feature_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({input_flat.size()}),
                                        &label_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            1, TensorShape({input_flat.size(), num_features_}),
                            &feature_tensor));

    auto label = label_tensor->flat<int64>();
    auto feature = feature_tensor->matrix<T>();
    for (int i = 0; i < input_flat.size(); ++i) {
      std::vector<string> entries =
          str_util::Split(input_flat(i), " ", str_util::SkipEmpty());
      OP_REQUIRES(ctx, (entries.size() > 0),
                  errors::InvalidArgument("No entries found for input[", i,
                                          "]: \"", input_flat(i), "\""));
      int64 label_value;
      OP_REQUIRES(
          ctx, strings::safe_strto64(entries[0].c_str(), &label_value),
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
        T feature_value;
        OP_REQUIRES(
            ctx, Convert(pair[1], &feature_value),
            errors::InvalidArgument("Feature format incorrect: ", entries[j]));
        feature(i, feature_index) = feature_value;
      }
    }
  }

 private:
  int64 num_features_;

  bool Convert(const string& s, T* value);
};

template <>
bool DecodeLibsvmOp<float>::Convert(const string& s, float* value) {
  return strings::safe_strtof(s.c_str(), value);
}
template <>
bool DecodeLibsvmOp<double>::Convert(const string& s, double* value) {
  return strings::safe_strtod(s.c_str(), value);
}
template <>
bool DecodeLibsvmOp<int32>::Convert(const string& s, int32* value) {
  return strings::safe_strto32(s.c_str(), value);
}
template <>
bool DecodeLibsvmOp<int64>::Convert(const string& s, int64* value) {
  return strings::safe_strto64(s.c_str(), value);
}

#define REGISTER_KERNEL(type)                                                \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DecodeLibsvm").Device(DEVICE_CPU).TypeConstraint<type>("dtype"), \
      DecodeLibsvmOp<type>);

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
#undef REGISTER_KERNEL

}  // namespace tensorflow
