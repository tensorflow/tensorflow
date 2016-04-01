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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchBatchNormTraining;

template <typename T>
struct LaunchBatchNormTraining<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const Tensor& input_param,
                     const Tensor& scale_param, const Tensor& bias_param,
                     Tensor* output) {
    std::cout << "Launched the BatchNormKernel??" << std::endl;

  }
};

template <typename T>
struct LaunchBatchNormTraining<CPUDevice, T> {
  static void launch(OpKernelContext* ctx, const Tensor& input_param,
                     const Tensor& scale_param, const Tensor& bias_param,
                     Tensor* output) {
    std::cout << "Launched the BatchNormKernel??" << std::endl;

  }
};

template <typename Device, typename T>
class BatchNormTrainingOp : public OpKernel {
  public:
    explicit BatchNormTrainingOp(OpKernelConstruction* context) : OpKernel(context) {
      const DataType dt = DataTypeToEnum<T>::v();
      OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt}));

      //Do some type checking here
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    }

    void Compute(OpKernelContext* context) override {
      //TODO a whole bunch of error checking
      const Tensor& input = context->input(0);
      const Tensor& scale = context->input(1);
      const Tensor& bias = context->input(2);

      Tensor* output = nullptr;
      TensorShape out_shape = input.shape();
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

      //TODO support other dimentions
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimentional", input.shape().DebugString()));

      LaunchBatchNormTraining<Device, T>::launch(
          context, input, scale, bias, output);
    }

  private:
    float epsilon_;

    TF_DISALLOW_COPY_AND_ASSIGN(BatchNormTrainingOp);
};

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("BatchNormTraining").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    BatchNormTrainingOp<GPUDevice, float>);

#endif // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("BatchNormTraining").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    BatchNormTrainingOp<CPUDevice, float>);

} // namespace tensorflow
