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

// See docs in ../ops/array_ops.cc.

#include <limits>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// --------------------------------------------------------------------------
template <typename Device, typename T>
class PackOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit PackOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* c) override {
    const int num = num_inputs();
    const Tensor& first_input = c->input(0);

    int expanded_num_dims = first_input.dims() + 1;
    int axis = axis_;
    if (axis < 0) axis += expanded_num_dims;

    OP_REQUIRES(c, 0 <= axis && axis < expanded_num_dims,
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -expanded_num_dims, ", ",
                                        expanded_num_dims, ")"));

    TensorShape output_shape(first_input.shape());
    output_shape.InsertDim(axis, num);

    // In the num = 1 case, just reshape the input
    if (num == 1) {
      Tensor output;
      CHECK(output.CopyFrom(first_input, output_shape));
      c->set_output(0, output);
      return;
    }

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));

    int64_t before_dim = 1;
    for (int i = 0; i < axis; ++i) {
      before_dim *= output_shape.dim_size(i);
    }

    int64_t after_dim = 1;
    for (int i = axis + 1; i < output_shape.dims(); ++i) {
      after_dim *= output_shape.dim_size(i);
    }

    const int64_t axis_dim = output_shape.dim_size(axis);

    const int64_t output_size = output->NumElements();
    auto output_flat = output->shaped<T, 2>({before_dim, after_dim * axis_dim});

    // Except for shapes, pack is a special case of concat, so we reuse the
    // same computational kernels.
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(num);
    for (int i = 0; i < num; ++i) {
      const Tensor& input = c->input(i);
      OP_REQUIRES(c, first_input.shape().IsSameSize(input.shape()),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: values[0].shape = ",
                      first_input.shape().DebugString(), " != values[", i,
                      "].shape = ", input.shape().DebugString()));

      inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
          input.shaped<T, 2>({before_dim, after_dim})));
    }
    if (output_size > 0) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      if (std::is_same<Device, GPUDevice>::value) {
        ConcatGPU<T>(c, inputs_flat, output, &output_flat);
        return;
      }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
    }
  }

 private:
  int axis_;
};

#define REGISTER_PACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      PackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_PACK);
TF_CALL_QUANTIZED_TYPES(REGISTER_PACK);
TF_CALL_qint16(REGISTER_PACK);
TF_CALL_quint16(REGISTER_PACK);

#if defined(IS_MOBILE_PLATFORM) && !defined(SUPPORT_SELECTIVE_REGISTRATION)
// Primarily used for SavedModel support on mobile.
REGISTER_PACK(tstring);
#endif  // defined(IS_MOBILE_PLATFORM) &&
        // !defined(SUPPORT_SELECTIVE_REGISTRATION)

#undef REGISTER_PACK

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PackOp<GPUDevice, type>)

TF_CALL_int64(REGISTER_GPU);
TF_CALL_int16(REGISTER_GPU);
TF_CALL_uint32(REGISTER_GPU);
TF_CALL_uint64(REGISTER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU);
#undef REGISTER_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Pack")
                            .Device(DEVICE_GPU)
                            .HostMemory("values")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PackOp<CPUDevice, int32>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
