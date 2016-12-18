/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include "sparsemax_loss.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SparsemaxLossOp : public OpKernel {
 public:
  explicit SparsemaxLossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& logits_in = context->input(0);
    const Tensor& sparsemax_in = context->input(1);
    const Tensor& labels_in = context->input(2);

    OP_REQUIRES(context,
                logits_in.IsSameSize(sparsemax_in),
                errors::InvalidArgument(
                    "logits and sparsemax must be same size: logits_size=",
                    logits_in.shape().DebugString(), " labels_size=",
                    sparsemax_in.shape().DebugString()));

    OP_REQUIRES(context,
                logits_in.IsSameSize(labels_in),
                errors::InvalidArgument(
                    "logits and labels must be same size: logits_size=",
                    logits_in.shape().DebugString(), " labels_size=",
                    labels_in.shape().DebugString()));

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));

    // Create an output tensor (vector with batch_size elements)
    Tensor* loss_out = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({logits_in.dim_size(0)}),
                   &loss_out));

    // Setup data view
    typename TTypes<T>::ConstMatrix logits = logits_in.matrix<T>();
    typename TTypes<T>::ConstMatrix sparsemax = sparsemax_in.matrix<T>();
    typename TTypes<T>::ConstMatrix labels = labels_in.matrix<T>();
    typename TTypes<T>::Vec losses = loss_out->flat<T>();

    // This will call the Eigen code.
    const Device& eigen_device = context->eigen_device<Device>();
    functor::SparsemaxLoss<Device, T>()(
      eigen_device, logits, sparsemax, labels, losses
    );
  }
};

// This will compile the Op for CPUDevice and compile the corresponding
// Eigen code.
#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(                     \
    Name("SparsemaxLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    SparsemaxLossOp<CPUDevice, T>);

TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU

#if GOOGLE_CUDA

// This will specify the symbol signature and tell the compiler that
// SparsemaxLoss<GPUDevice, T> will be compiled from another file.
namespace functor {
#define DECLARE_GPU_SPEC(T)                           \
  template <>                                         \
  void SparsemaxLoss<GPUDevice, T>::operator()(       \
    const GPUDevice& d,                               \
    typename TTypes<T>::ConstMatrix logits,           \
    typename TTypes<T>::ConstMatrix sparsemax,        \
    typename TTypes<T>::ConstMatrix labels,           \
    typename TTypes<T>::Vec losses);                  \
  extern template struct SparsemaxLoss<GPUDevice, T>;

TF_CALL_half(DECLARE_GPU_SPEC);
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_double(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// This will compile the Op for GPUDevice but **not** compile the corresponding
// Eigen code.
#define REGISTER_GPU(T) REGISTER_KERNEL_BUILDER(                     \
    Name("SparsemaxLoss").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
    SparsemaxLossOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
