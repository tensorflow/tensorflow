/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#if defined(INTEL_MKL)
#define EIGEN_USE_THREADS

#include <math.h>

#include <limits>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class MklRequantizationRangePerChannelOp : public OpKernel {
 public:
  explicit MklRequantizationRangePerChannelOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("clip_value_max", &clip_value_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(kInputTensorIndex);
    const Tensor& input_min = ctx->input(kInputMinIndex);
    const Tensor& input_max = ctx->input(kInputMaxIndex);

    const size_t depth = input_max.NumElements();
    OP_REQUIRES(
        ctx, input_min.dim_size(0) == depth,
        errors::InvalidArgument("input_min has incorrect size, expected ",
                                depth, " was ", input_min.dim_size(0)));
    OP_REQUIRES(
        ctx, input_max.dim_size(0) == depth,
        errors::InvalidArgument("input_max has incorrect size, expected ",
                                depth, " was ", input_max.dim_size(0)));
    OP_REQUIRES(
        ctx, input_min.NumElements() == depth,
        errors::InvalidArgument("input_min must have the same number of "
                                "elements as input_max, got ",
                                input_min.NumElements(), " and ", depth));
    OP_REQUIRES(ctx, input.NumElements() > 0,
                errors::InvalidArgument("input must not be empty"));
    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("input must be in NHWC format"));
    OP_REQUIRES(
        ctx, input.dim_size(3) == depth,
        errors::InvalidArgument(
            "input must have same number of channels as length of input_min: ",
            input.dim_size(3), " vs ", depth));

    const float* input_min_data = input_min.flat<float>().data();
    const float* input_max_data = input_max.flat<float>().data();
    std::vector<float> ranges(depth);
    bool is_non_negative = true;
    Eigen::array<int, 2> shuffling({1, 0});
    auto input_matrix = input.flat_inner_dims<qint32>();

    // TODO(intel-tf): Verify performance of not transposing and finding min max
    // directly from input_matrix vs the one presented below of transposing and
    // using the transposed matrix as the transposing operation in itself might
    // be more costly.
    // Note that this operation is a calibration step for quantization and will
    // cease to exist in the final inference graph(will exist as a const node).
    auto transposed_input = input_matrix.shuffle(shuffling);

    // Find the ranges of each channel in parallel.
    float out_min_max = std::numeric_limits<float>::min();

#ifdef ENABLE_ONEDNN_OPENMP
#ifdef _MSC_VER
#pragma omp parallel for
#else
#pragma omp parallel for reduction(max : out_min_max)
#endif
#endif  // ENABLE_ONEDNN_OPENMP
    // TODO(intel-tf): Add eigen parallel_for
    for (int64_t i = 0; i < depth; ++i) {
      Eigen::Tensor<qint32, 0, Eigen::RowMajor> min =
          transposed_input.chip<0>(i).minimum();
      Eigen::Tensor<qint32, 0, Eigen::RowMajor> max =
          transposed_input.chip<0>(i).maximum();
      const int32_t min_per_channel = min();
      const int32_t max_per_channel = max();
      const int32_t abs_max =
          std::max(std::abs(min_per_channel), std::abs(max_per_channel));
      float scale =
          std::max(std::abs(input_min_data[i]), std::abs(input_max_data[i]));
      ranges[i] =
          scale * static_cast<float>(abs_max) / static_cast<float>(1L << 31);
      if (min_per_channel < 0) is_non_negative = false;

      // Thread-local out_min_max.
      out_min_max = std::max(out_min_max, ranges[i]);
    }

    // All local out_min_max gets max-reduced into one global out_min_max at
    // the end of the loop by specifying reduction(max:out_min_max) along with
    // omp parallel for.

    // Fixing max to clip_value_max_ (example 6.0 to support relu6)
    if (out_min_max > clip_value_max_) out_min_max = clip_value_max_;

    Tensor* output_min = nullptr;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(kOutputMinIndex, {}, &output_min));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(kOutputMaxIndex, {}, &output_max));
    output_min->scalar<float>()() = is_non_negative ? 0.0f : -out_min_max;
    output_max->scalar<float>()() = out_min_max;
  }

 private:
  float clip_value_max_ = std::numeric_limits<float>::infinity();
  const int kInputTensorIndex = 0;
  const int kInputMinIndex = 1;
  const int kInputMaxIndex = 2;
  const int kOutputMinIndex = 0;
  const int kOutputMaxIndex = 1;
};

REGISTER_KERNEL_BUILDER(Name("RequantizationRangePerChannel")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint32>("T"),
                        MklRequantizationRangePerChannelOp);
}  // namespace tensorflow
#endif  // INTEL_MKL
