#define EIGEN_USE_THREADS
#include "tensorflow/contrib/naturali/kernels/lookahead_grad_ops.h"
#include <cuda_runtime.h>

using namespace tensorflow;

template<typename T>
__global__ void kernel_grad_input(int dim_tau, const T* filter, const T* output_grad, T* output) {
  int dim_timestep = gridDim.x;
  int dim_batch = gridDim.y;
  int dim_feature = blockDim.x;
  int timestep = blockIdx.x;
  int batch = blockIdx.y;
  int feature = threadIdx.x;
  output[(timestep * dim_batch + batch) * dim_feature + feature] = 0;
  for(int input_begin = 0; input_begin < dim_tau; input_begin++) {
    int index = input_begin + timestep - dim_tau + 1;
    int filter_idx = dim_tau - 1 - input_begin;
    if (index >= 0 && filter_idx >= 0) {
      output[(timestep * dim_batch + batch) * dim_feature + feature] += output_grad[(index * dim_batch + batch) * dim_feature + feature] * filter[filter_idx * dim_feature + feature];
    }
  }
}

template<typename T>
__global__ void kernel_grad_filter(int dim_batch, int dim_timestep, const T* input, const T* output_grad, T* output) {
  int dim_feature = blockDim.x;
  int dim_tau = gridDim.x;
  int feature = threadIdx.x;
  int tau = blockIdx.x;
  output[tau * dim_feature + feature] = 0;
  for (int batch = 0; batch < dim_batch; batch++) {
    for (int timestep = 0; timestep < dim_timestep - tau; timestep++) {
      output[tau * dim_feature + feature] += output_grad[(timestep * dim_batch + batch) * dim_feature + feature] * input[((timestep + tau) * dim_batch + batch) * dim_feature + feature];
    }
  }
}

template<typename T>
class LookaheadGradOp<T, 1> : public OpKernel {
 public:
  explicit LookaheadGradOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt, dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& output_grad_tensor = context->input(2);
    auto output_grad = output_grad_tensor.tensor<T, 3>();

    // Check that dimension is equal
    OP_REQUIRES(
        context, input_tensor.dim_size(2) == filter_tensor.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));
    OP_REQUIRES(
        context, (input_tensor.dim_size(0) == output_grad_tensor.dim_size(0)) &&
                 (input_tensor.dim_size(1) == output_grad_tensor.dim_size(1)) &&
                 (input_tensor.dim_size(2) == output_grad_tensor.dim_size(2)),
        errors::InvalidArgument("input's dimensions and output_grad's dimensions are not equal"));

    // Create input grad output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    int dim_timestep = input_tensor.dim_size(0);
    int dim_batch = input_tensor.dim_size(1);
    int dim_feature = input_tensor.dim_size(2);
    int dim_tau = filter_tensor.dim_size(0);

    dim3 grid(dim_timestep, dim_batch);
    auto device = context->template eigen_device<Eigen::GpuDevice>();
    auto stream = device.stream();
    kernel_grad_input<T><<<grid, dim_feature, 0, stream>>>(dim_tau, &filter(0, 0), &output_grad(0, 0, 0), &output(0, 0, 0));
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();

    kernel_grad_filter<T><<<dim_tau, dim_feature, 0, stream>>>(dim_batch, dim_timestep, &input(0, 0, 0), &output_grad(0, 0, 0), &output2(0, 0));
  }
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), LookaheadGradOp<float, 1>);

