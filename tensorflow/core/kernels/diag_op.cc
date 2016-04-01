/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
template <typename T, size_t NumDims, size_t DoubleNumDims>
class DiagonalGenerator {
 public:
  explicit DiagonalGenerator(const Tensor& diagonal) : diagonal_(diagonal) {
    static_assert(DoubleNumDims == 2 * NumDims,
                  "The second size must be the double of the first size.");
    CHECK_EQ(diagonal.dims(), NumDims);
  }
  T operator()(
      const Eigen::array<Eigen::DenseIndex, DoubleNumDims>& coordinates) const {
    Eigen::array<Eigen::DenseIndex, NumDims> index;
    for (size_t i = 0; i < NumDims; ++i) {
      if (coordinates[i] != coordinates[NumDims + i]) {
        return T(0);
      }
      index[i] = coordinates[i];
    }
    return diagonal_.tensor<T, NumDims>()(index);
  }

 private:
  Tensor diagonal_;
};

template <typename T, size_t NumDims>
class DiagonalExtractor {
 public:
  explicit DiagonalExtractor(const Tensor& tensor) : tensor_(tensor) {
    CHECK_EQ(tensor.dims(), 2 * NumDims);
  }
  T operator()(const Eigen::array<Eigen::Index, NumDims>& coordinates) const {
    Eigen::array<Eigen::Index, 2 * NumDims> index;
    for (size_t j = 0; j < NumDims; ++j){
      index[j] = coordinates[j];
    }
    for (size_t j = NumDims; j < 2 * NumDims; ++j){
      index[j] = index[j - NumDims];
    }
    return tensor_.tensor<T, 2 * NumDims>()(index);
  }

 private:
  Tensor tensor_;
};
  
}  // namespace

// Generate the diagonal tensor with the diagonal set to the input tensor.
// It only allows up to rank 3 input tensor, so the output tensor is up to
// rank 6.
template <typename T>
class DiagOp : public OpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& diagonal = context->input(0);
    const int num_dims = diagonal.dims();
    OP_REQUIRES(context, 1 <= num_dims && num_dims <= 3,
                errors::InvalidArgument("Expected 1 <= dims <= 3, got shape ",
                                        diagonal.shape().DebugString()));
    TensorShape out_shape;
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    for (int i = 0; i < num_dims; ++i) {
      out_shape.AddDim(diagonal.dim_size(i));
    }
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_tensor));
    switch (num_dims) {
      case 1:
        output_tensor->tensor<T, 2>() = output_tensor->tensor<T, 2>().generate(
            DiagonalGenerator<T, 1, 2>(diagonal));
        break;
      case 2:
        output_tensor->tensor<T, 4>() = output_tensor->tensor<T, 4>().generate(
            DiagonalGenerator<T, 2, 4>(diagonal));
        break;
      case 3:
        output_tensor->tensor<T, 6>() = output_tensor->tensor<T, 6>().generate(
            DiagonalGenerator<T, 3, 6>(diagonal));
        break;
      default:
        context->SetStatus(errors::Unimplemented(
            "Diagonal of rank ", num_dims, " tensor is not supported yet."));
        return;
    }
  }
};

#define REGISTER_DIAGOP(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("Diag").Device(DEVICE_CPU).TypeConstraint<T>("T"), DiagOp<T>)

REGISTER_DIAGOP(double);
REGISTER_DIAGOP(float);
REGISTER_DIAGOP(int32);
REGISTER_DIAGOP(int64);

#undef REGISTER_DIAGOP


// Generate the diagonal tensor with the diagonal set to the input tensor.
// It only allows rank 2, 4, or 6 input tensor, so the output tensor is 
// rank 1, 2, or 3.
template <typename T>
class DiagPartOp : public OpKernel {
 public:
  explicit DiagPartOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(context, 2 == num_dims || 4 == num_dims || 6 == num_dims, 
                errors::InvalidArgument("The rank of the tensor should be 2, \
                                         4, or 6, got shape ",
                                        tensor.shape().DebugString()));
    for (int i = 0; i < out_dims; i++){
      OP_REQUIRES(context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
                  errors::InvalidArgument(
                    "Invalid shape ", tensor.shape().DebugString(),
                    ": dimensions ", i, " and ", i + out_dims, " do not match.")
                  );
    }

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      out_shape.AddDim(tensor.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output));

    switch (num_dims) {
      case 2:
        output->tensor<T, 1>() = output->tensor<T, 1>().generate(
          DiagonalExtractor<T, 1>(tensor));
        break; 
      case 4:
        output->tensor<T, 2>() = output->tensor<T, 2>().generate(
          DiagonalExtractor<T, 2>(tensor));
        break;
      case 6:
        output->tensor<T, 3>() = output->tensor<T, 3>().generate(
          DiagonalExtractor<T, 3>(tensor));
        break;      
      default:
        context->SetStatus(errors::Unimplemented(
          "Diagonal of rank ", num_dims, " tensor is not supported yet."));
        return;
    }
  }
};

#define REGISTER_DIAGPARTOP(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("DiagPart").Device(DEVICE_CPU).TypeConstraint<T>("T"), DiagPartOp<T>)

REGISTER_DIAGPARTOP(double);
REGISTER_DIAGPARTOP(float);
REGISTER_DIAGPARTOP(int32);
REGISTER_DIAGPARTOP(int64);

#undef REGISTER_DIAGPARTOP
  
}  // namespace tensorflow
