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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/data_format_ops.h"

#include <map>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Ensure that `src` and `dst` define a valid permutation.
// Ops defined in this file assume that user specifies a permutation via two
// string attributes. This check validates that these attributes properly define
// it to prevent security vulnerabilities.
static bool IsValidPermutation(const std::string& src, const std::string& dst) {
  if (src.size() != dst.size()) {
    return false;
  }

  std::map<char, bool> characters;

  // Every character in `src` must be present only once
  for (const auto c : src) {
    if (characters[c]) {
      return false;
    }
    characters[c] = true;
  }

  // Every character in `dst` must show up in `src` exactly once
  for (const auto c : dst) {
    if (!characters[c]) {
      return false;
    }
    characters[c] = false;
  }

  // At this point, characters[] has been switched to true and false exactly
  // once for all character in `src` (and `dst`) so we have a valid permutation
  return true;
}

template <typename Device, typename T>
class DataFormatDimMapOp : public OpKernel {
 public:
  explicit DataFormatDimMapOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string src_format;
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format));
    string dst_format;
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(context, src_format.size() == 4 || src_format.size() == 5,
                errors::InvalidArgument(
                    "Source format must be of length 4 or 5, received "
                    "src_format = ",
                    src_format));
    OP_REQUIRES(context, dst_format.size() == 4 || dst_format.size() == 5,
                errors::InvalidArgument("Destination format must be of length "
                                        "4 or 5, received dst_format = ",
                                        dst_format));
    OP_REQUIRES(
        context, IsValidPermutation(src_format, dst_format),
        errors::InvalidArgument(
            "Destination and source format must determine a permutation, got ",
            src_format, " and ", dst_format));
    dst_idx_ = Tensor(DT_INT32, {static_cast<int64>(src_format.size())});
    for (int i = 0; i < src_format.size(); ++i) {
      for (int j = 0; j < dst_format.size(); ++j) {
        if (dst_format[j] == src_format[i]) {
          dst_idx_.vec<int>()(i) = j;
          break;
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    functor::DataFormatDimMap<Device, T>()(context->eigen_device<Device>(),
                                           input.flat<T>(), output->flat<T>(),
                                           dst_idx_.vec<int>());
  }

  Tensor dst_idx_;
};

template <typename Device, typename T>
class DataFormatVecPermuteOp : public OpKernel {
 public:
  explicit DataFormatVecPermuteOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string src_format;
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format));
    OP_REQUIRES(context, src_format.size() == 4 || src_format.size() == 5,
                errors::InvalidArgument(
                    "Source format must be of length 4 or 5, received "
                    "src_format = ",
                    src_format));
    string dst_format;
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(context, dst_format.size() == 4 || dst_format.size() == 5,
                errors::InvalidArgument("Destination format must be of length "
                                        "4 or 5, received dst_format = ",
                                        dst_format));
    OP_REQUIRES(
        context, IsValidPermutation(src_format, dst_format),
        errors::InvalidArgument(
            "Destination and source format must determine a permutation, got ",
            src_format, " and ", dst_format));
    src_format_ = src_format;
    dst_format_ = dst_format;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 1 || input.dims() == 2,
                errors::InvalidArgument(
                    "input must be a vector or 2D tensor, but got shape ",
                    input.shape().DebugString()));
    if (input.dims() == 1) {
      OP_REQUIRES(context,
                  input.NumElements() == 2 || input.NumElements() == 4 ||
                      input.NumElements() == 5,
                  errors::InvalidArgument(
                      "1D input must be of size 2, 4 or 5, but got shape ",
                      input.shape().DebugString()));
    } else if (input.dims() == 2) {
      OP_REQUIRES(context, input.dim_size(0) == 2 || input.dim_size(0) == 4,
                  errors::InvalidArgument("First dimension of 2D input must be "
                                          "of size 2 or 4, but got shape ",
                                          input.shape().DebugString()));
      OP_REQUIRES(
          context, input.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input.shape().DebugString()));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    // Support 1D and 2D cases.
    Eigen::DSizes<Eigen::DenseIndex, 8> dst_idx;
    string src_format_str = src_format_;
    string dst_format_str = dst_format_;
    if (input.dim_size(0) == 2) {
      // If the input is a vector of size 2, treat the two elements as spatial
      // dimensions.
      auto keep_only_spatial_dimensions = [](string* format_str) -> void {
        auto new_end = std::remove_if(
            format_str->begin(), format_str->end(),
            [](const char dim) { return dim != 'H' && dim != 'W'; });
        format_str->erase(new_end, format_str->end());
      };
      keep_only_spatial_dimensions(&src_format_str);
      keep_only_spatial_dimensions(&dst_format_str);
      OP_REQUIRES(context,
                  src_format_str.size() == 2 && dst_format_str.size() == 2,
                  errors::InvalidArgument(
                      "Format specifier must contain H and W for 2D case"));
    }
    ComputeDstIndex(src_format_str, dst_format_str, input.dims(), &dst_idx);

    functor::DataFormatVecPermute<Device, T>()(context->eigen_device<Device>(),
                                               input.flat<T>(),
                                               output->flat<T>(), dst_idx);
  }

 private:
  // Finds out the destination index. Support 1D and 2D cases.
  // Example: HWNC --> NHWC
  // 1D: dst = [1, 2, 0, 3],
  // 2D: dst = [2, 3, 4, 5, 0, 1, 6, 7]
  static void ComputeDstIndex(const string& src_format_str,
                              const string& dst_format_str, int num_dim,
                              Eigen::DSizes<Eigen::DenseIndex, 8>* dst) {
    for (int i = 0; i < src_format_str.size(); ++i) {
      for (int j = 0; j < dst_format_str.size(); ++j) {
        if (dst_format_str[j] != src_format_str[i]) continue;
        // Found the dst index. Set output based on the number of dims.
        for (int k = 0; k < num_dim; ++k) {
          (*dst)[i * num_dim + k] = j * num_dim + k;
        }
      }
    }
  }

  string src_format_;
  string dst_format_;
};

#define REGISTER_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DataFormatDimMapOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DataFormatVecPermute").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DataFormatVecPermuteOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                             \
  REGISTER_KERNEL_BUILDER(Name("DataFormatDimMap")     \
                              .Device(DEVICE_CPU)      \
                              .Label("host")           \
                              .TypeConstraint<T>("T"), \
                          DataFormatDimMapOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                             \
  REGISTER_KERNEL_BUILDER(Name("DataFormatVecPermute") \
                              .Device(DEVICE_CPU)      \
                              .Label("host")           \
                              .TypeConstraint<T>("T"), \
                          DataFormatVecPermuteOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                    \
  template <>                                                  \
  void DataFormatDimMap<GPUDevice, T>::operator()(             \
      const GPUDevice& d, typename TTypes<T>::ConstFlat x,     \
      typename TTypes<T>::Flat y, const TTypes<int>::Vec dst); \
  extern template struct DataFormatDimMap<GPUDevice, T>;
#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC

#define DECLARE_GPU_SPEC(T)                                \
  template <>                                              \
  void DataFormatVecPermute<GPUDevice, T>::operator()(     \
      const GPUDevice& d, typename TTypes<T>::ConstFlat x, \
      typename TTypes<T>::Vec y,                           \
      const Eigen::DSizes<Eigen::DenseIndex, 8>& dst_idx); \
  extern template struct DataFormatVecPermute<GPUDevice, T>;
#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);
TF_CALL_int32(DECLARE_GPU_SPECS);
TF_CALL_int64(DECLARE_GPU_SPECS);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DataFormatDimMap").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatDimMapOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("DataFormatDimMap")                        \
                              .Device(DEVICE_GPU)                         \
                              .HostMemory("x")                            \
                              .HostMemory("y")                            \
                              .Label("host")                              \
                              .TypeConstraint<T>("T"),                    \
                          DataFormatDimMapOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#define REGISTER_GPU_KERNEL(T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("DataFormatVecPermute").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DataFormatVecPermuteOp<GPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("DataFormatVecPermute")                        \
                              .Device(DEVICE_GPU)                             \
                              .HostMemory("x")                                \
                              .HostMemory("y")                                \
                              .Label("host")                                  \
                              .TypeConstraint<T>("T"),                        \
                          DataFormatVecPermuteOp<CPUDevice, T>);
TF_CALL_int32(REGISTER_GPU_KERNEL);
TF_CALL_int64(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
