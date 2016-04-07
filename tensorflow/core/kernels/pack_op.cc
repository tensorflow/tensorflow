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
typedef Eigen::GpuDevice GPUDevice;

// --------------------------------------------------------------------------
template <typename Device, typename T>
class PackOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit PackOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int num = values.size();

    // Verify that all input shapes match
    for (int i = 1; i < num; i++) {
      OP_REQUIRES(c, values[0].shape().IsSameSize(values[i].shape()),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: values[0].shape = ",
                      values[0].shape().DebugString(), " != values[", i,
                      "].shape = ", values[i].shape().DebugString()));
    }
    TensorShape output_shape(values[0].shape());
    output_shape.InsertDim(0, num);

    // In the num = 1 case, just reshape the input
    if (num == 1) {
      Tensor output;
      CHECK(output.CopyFrom(values[0], output_shape));
      c->set_output(0, output);
      return;
    }

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));

    const int64 output_size = output->NumElements();
    if (output_size > 0) {
      auto output_flat = output->shaped<T, 2>({1, output_size});

      // Except for shapes, pack is a special case of concat, so we reuse the
      // same computational kernels.
      ConstMatrixVector inputs_flat;
      inputs_flat.reserve(num);
      for (int i = 0; i < num; ++i) {
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            values[i].shaped<T, 2>({1, values[i].NumElements()})));
      }
      if (std::is_same<Device, GPUDevice>::value) {
        // Switching indexing to int64 might cause performance issues.
        // Hence, we keep int32 indexing in the GPU kernel unless we need to
        // switch to int64.
        if (output_size < std::numeric_limits<int32>::max()) {
          ConcatGPU32<T>(c->eigen_gpu_device(), inputs_flat, &output_flat);
        } else {
          ConcatGPU64<T>(c->eigen_gpu_device(), inputs_flat, &output_flat);
        }
      } else {
        ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
      }
    }
  }
};

#define REGISTER_PACK(type)                                      \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      PackOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_PACK);
REGISTER_PACK(quint8);
REGISTER_PACK(qint8);
REGISTER_PACK(qint32);
REGISTER_PACK(bfloat16);

#undef REGISTER_PACK

#if GOOGLE_CUDA

#define REGISTER_GPU(type)                                       \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Pack").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PackOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
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

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
