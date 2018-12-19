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

// See docs in ../ops/array_ops.cc.
#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include <math.h>
#include <limits>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class MklRequantizationRangePerChannelOp : public OpKernel {
 public:
  explicit MklRequantizationRangePerChannelOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("clip_value_max", &clip_value_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = MklGetInput(ctx, kInputTensorIndex);
    const Tensor& input_min = MklGetInput(ctx, this->kInputMin);
    const Tensor& input_max = MklGetInput(ctx, this->kInputMax);

    size_t depth = input_max.NumElements();
    OP_REQUIRES(ctx, input_min.dim_size(0) == depth,
                errors::InvalidArgument("min has incorrect size, expected ",
                                        depth, " was ", input_min.dim_size(0)));
    OP_REQUIRES(ctx, input_max.dim_size(0) == depth,
                errors::InvalidArgument("max has incorrect size, expected ",
                                        depth, " was ", input_max.dim_size(0)));

    const float* input_min_data = input_min.flat<float>().data();
    const float* input_max_data = input_max.flat<float>().data();
    std::vector<float> ranges(depth);
    bool is_non_negative = true;
    Eigen::array<int, 2> shuffling({1, 0});
    auto input_matrix = input.flat_inner_dims<qint32>();
    auto transposed_input = input_matrix.shuffle(shuffling);

#pragma omp parallel for
    for (size_t i = 0; i < depth; i++) {
      Eigen::Tensor<qint32, 0, Eigen::RowMajor> min =
          transposed_input.chip<0>(i).minimum();
      Eigen::Tensor<qint32, 0, Eigen::RowMajor> max =
          transposed_input.chip<0>(i).maximum();
      int32_t min_per_channel = min();
      int32_t max_per_channel = max();
      int32_t abs_max =
          std::max(std::abs(min_per_channel), std::abs(max_per_channel));
      float scale =
          std::max(std::abs(input_min_data[i]), std::abs(input_max_data[i]));
      ranges[i] = (scale * (float)abs_max / (float)(1L << 31));
      if (min_per_channel < 0) is_non_negative = false;
    }

    float out_min_max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < depth; i++) {
      if (out_min_max < ranges[i]) out_min_max = ranges[i];
    }
    // Fixing max to clip_value_max_ (example 6.0 to support relu6)
    if (out_min_max > clip_value_max_) out_min_max = clip_value_max_;

    Tensor* output_min = nullptr;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(kOutputMin, {}, &output_min));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(kOutputMax, {}, &output_max));
    output_min->flat<float>()(0) = is_non_negative ? 0.0f : out_min_max * -1.0f;
    output_max->flat<float>()(0) = out_min_max;
  }

 private:
  float clip_value_max_ = std::numeric_limits<float>::infinity();
  const int kInputTensorIndex = 0;
  const int kInputMin = 1;
  const int kInputMax = 2;
  const int kOutputMin = 0;
  const int kOutputMax = 1;
};

REGISTER_KERNEL_BUILDER(Name("RequantizationRangePerChannel")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T"),
                        MklRequantizationRangePerChannelOp);
}  // namespace tensorflow
#endif  // INTEL_MKL
