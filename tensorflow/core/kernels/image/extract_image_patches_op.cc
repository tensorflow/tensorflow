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

// See docs in ../ops/image_ops.cc.

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image/extract_image_patches_op.h"
#include "tensorflow/core/util/overflow.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

static inline void ParseAttributeVec4(OpKernelConstruction* context,
                                      const std::string& attr_name,
                                      std::vector<int32_t>* attr) {
  OP_REQUIRES_OK(context, context->GetAttr(attr_name, attr));
  OP_REQUIRES(context, (*attr)[0] == 1 && (*attr)[3] == 1,
              absl::UnimplementedError(
                  absl::StrCat("Only support ", attr_name, " across space.")));
  OP_REQUIRES(
      context, (*attr)[1] >= 1 && (*attr)[2] >= 1,
      absl::OutOfRangeError(absl::StrCat(attr_name, " is out of range.")));
}

template <typename Device, typename T>
class ExtractImagePatchesOp : public UnaryOp<T> {
 public:
  explicit ExtractImagePatchesOp(OpKernelConstruction* context)
      : UnaryOp<T>(context) {
    ParseAttributeVec4(context, "ksizes", &ksizes_);
    ParseAttributeVec4(context, "strides", &strides_);
    ParseAttributeVec4(context, "rates", &rates_);
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, channels ]
    const Tensor& input = context->input(0);
    OP_REQUIRES(
        context, input.dims() == 4,
        absl::InvalidArgumentError(absl::StrCat("input must be 4-dimensional",
                                                input.shape().DebugString())));

    const int64_t batch = input.dim_size(0);
    const int64_t in_rows = input.dim_size(1);
    const int64_t in_cols = input.dim_size(2);
    const int64_t depth = input.dim_size(3);

    const int64_t ksize_rows = ksizes_[1];
    const int64_t ksize_cols = ksizes_[2];

    const int64_t stride_rows = strides_[1];
    const int64_t stride_cols = strides_[2];

    const int64_t rate_rows = rates_[1];
    const int64_t rate_cols = rates_[2];

    const int64_t ksize_rows_eff =
        ksize_rows + (ksize_rows - 1) * (rate_rows - 1);
    const int64_t ksize_cols_eff =
        ksize_cols + (ksize_cols - 1) * (rate_cols - 1);

    int64_t out_rows = 0, out_cols = 0;
    int64_t pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context, GetWindowedOutputSize(
                                in_rows, ksize_rows_eff, /*dilation_rate=*/1,
                                stride_rows, padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSize(
                                in_cols, ksize_cols_eff, /*dilation_rate=*/1,
                                stride_cols, padding_, &out_cols, &pad_cols));

    int64_t patch_size = MultiplyWithoutOverflow(
        ksize_rows, MultiplyWithoutOverflow(ksize_cols, depth));
    OP_REQUIRES(context, patch_size >= 0,
                absl::InvalidArgumentError(
                    absl::StrCat("Output size would overflow: ", ksize_rows,
                                 " x ", ksize_cols, " x ", depth)));

    const std::vector<int64_t> out_sizes = {batch, out_rows, out_cols,
                                            patch_size};
    TensorShape out_shape(out_sizes);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    functor::ExtractImagePatchesForward<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 4>(),
        static_cast<int>(ksize_rows), static_cast<int>(ksize_cols),
        static_cast<int>(stride_rows), static_cast<int>(stride_cols),
        static_cast<int>(rate_rows), static_cast<int>(rate_cols),
        BrainPadding2EigenPadding(padding_), output->tensor<T, 4>());
  }

 private:
  std::vector<int32_t> ksizes_;
  std::vector<int32_t> strides_;
  std::vector<int32_t> rates_;

  Padding padding_;

  ExtractImagePatchesOp(const ExtractImagePatchesOp&) = delete;
  void operator=(const ExtractImagePatchesOp&) = delete;
};

// Registration of the CPU implementations.
#define REGISTER(T)                                                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ExtractImagePatches").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExtractImagePatchesOp<CPUDevice, T>);

TF_CALL_NUMBER_TYPES(REGISTER);

#undef REGISTER

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// Forward declarations of the functor specializations for GPU.
namespace functor {

#define DECLARE_GPU_SPEC(T)                                             \
  template <>                                                           \
  void ExtractImagePatchesForward<GPUDevice, T>::operator()(            \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,     \
      int patch_rows, int patch_cols, int stride_rows, int stride_cols, \
      int rate_rows, int rate_cols, const Eigen::PaddingType& padding,  \
      typename TTypes<T, 4>::Tensor output);                            \
  extern template struct ExtractImagePatchesForward<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER(T)                                                          \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ExtractImagePatches").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExtractImagePatchesOp<GPUDevice, T>);

TF_CALL_GPU_ALL_TYPES(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
