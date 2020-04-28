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

/*
See extract_image_patches_op* files and docs for extract_image_patches in
../ops/image_ops.cc.

Rates are not supported as of now, but the comments hint how to edit the code
when rates are to be added.
*/

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/extract_volume_patches_op.h"

#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

static inline void ParseAttributeVec5(OpKernelConstruction* context,
                                      const string& attr_name,
                                      std::vector<int32>* attr) {
  OP_REQUIRES_OK(context, context->GetAttr(attr_name, attr));
  OP_REQUIRES(
      context, (*attr)[0] == 1 && (*attr)[4] == 1,
      errors::Unimplemented("Only support ", attr_name, " across space."));
  OP_REQUIRES(context, (*attr)[1] >= 1 && (*attr)[2] >= 1 && (*attr)[3] >= 1,
              errors::OutOfRange(attr_name, " is out of range."));
}

template <typename Device, typename T>
class ExtractVolumePatchesOp : public UnaryOp<T> {
 public:
  explicit ExtractVolumePatchesOp(OpKernelConstruction* context)
      : UnaryOp<T>(context) {
    ParseAttributeVec5(context, "ksizes", &ksizes_);
    ParseAttributeVec5(context, "strides", &strides_);
    // ParseAttributeVec5(context, "rates", &rates_);
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_planes, in_rows, in_cols, channels ]
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 5,
                errors::InvalidArgument("input must be 5-dimensional",
                                        input.shape().DebugString()));

    const int batch = input.dim_size(0);
    const int in_planes = input.dim_size(1);
    const int in_rows = input.dim_size(2);
    const int in_cols = input.dim_size(3);
    const int depth = input.dim_size(4);

    const int ksize_planes = ksizes_[1];
    const int ksize_rows = ksizes_[2];
    const int ksize_cols = ksizes_[3];

    const int stride_planes = strides_[1];
    const int stride_rows = strides_[2];
    const int stride_cols = strides_[3];

    /*
    // TODO(hsgkim): enable rates
    // Rates are disabled as of now due to Eigen's definitions of
    // `extract_volume_patch` functions; none of them accept rates
    // as its argument and rates are fixed to (1, 1, 1, 1, 1). A
    // workaround has to be found for this.
    // In order to enable rates, uncomment the following lines and use
    // ksize_*_eff instead of ksize_* for the second argument of
    // GetWindowedOutputSize calls.

    const int rate_planes = rates_[1];
    const int rate_rows = rates_[2];
    const int rate_cols = rates_[3];

    const int ksize_planes_eff = ksize_planes +
                                 (ksize_planes - 1) * (rate_planes - 1);
    const int ksize_rows_eff = ksize_rows + (ksize_rows - 1) * (rate_rows - 1);
    const int ksize_cols_eff = ksize_cols + (ksize_cols - 1) * (rate_cols - 1);
    */

    int64 out_planes = 0, out_rows = 0, out_cols = 0;
    int64 pad_planes = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_planes, ksize_planes, stride_planes,
                                         padding_, &out_planes, &pad_planes));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_rows, ksize_rows, stride_rows,
                                         padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_cols, ksize_cols, stride_cols,
                                         padding_, &out_cols, &pad_cols));

    const std::vector<int64> out_sizes = {
        batch, out_planes, out_rows, out_cols,
        ksize_planes * ksize_rows * ksize_cols * depth};
    TensorShape out_shape(out_sizes);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    functor::ExtractVolumePatchesForward<Device, T>()(
        context->eigen_device<Device>(), input.tensor<T, 5>(), ksize_planes,
        ksize_rows, ksize_cols, stride_planes, stride_rows, stride_cols,
        /* rate_planes, rate_rows, rate_cols, */
        BrainPadding2EigenPadding(padding_), output->tensor<T, 5>());
  }

 private:
  std::vector<int32> ksizes_;
  std::vector<int32> strides_;
  // std::vector<int32> rates_;

  Padding padding_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExtractVolumePatchesOp);
};

// Registration of the CPU implementations.
#define REGISTER(T)                                                           \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ExtractVolumePatches").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExtractVolumePatchesOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {

// clang-format off
#define DECLARE_GPU_SPEC(T)                                         \
  template <>                                                       \
  void ExtractVolumePatchesForward<GPUDevice, T>::operator()(       \
      const GPUDevice& d, typename TTypes<T, 5>::ConstTensor input, \
      int patch_planes, int patch_rows, int patch_cols,             \
      int stride_planes, int stride_rows, int stride_cols,          \
      /* int rate_planes, int rate_rows, int rate_cols, */          \
      const Eigen::PaddingType& padding,                            \
      typename TTypes<T, 5>::Tensor output);                        \
  extern template struct ExtractVolumePatchesForward<GPUDevice, T>;
// clang-format on

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER(T)                                                           \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ExtractVolumePatches").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExtractVolumePatchesOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
