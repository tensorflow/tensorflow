#include "tensorflow/core/kernels/lookahead_ops.h"

#include <memory>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include <cuda_runtime.h>


using namespace tensorflow;

template<typename T>
__global__ void kernel(int dim_tau, const T* input, const T* filter, T* output) {
  int dim_timestep = gridDim.x;
  int dim_batch = gridDim.y;
  int dim_feature = blockDim.x;
  int timestep = blockIdx.x;
  int batch = blockIdx.y;
  int feature = threadIdx.x;
  output[(timestep * dim_batch + batch) * dim_feature + feature] = 0;
  for(int tau = 0; tau < dim_tau && timestep + tau < dim_timestep; tau++) {
    output[(timestep * dim_batch + batch) * dim_feature + feature] += input[((timestep + tau) * dim_batch + batch) * dim_feature + feature] * filter[tau * dim_feature + feature];
  }
}
template<typename T>
class LookaheadOp<T, 1> : public OpKernel {
 public:
  explicit LookaheadOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();
    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    // Check that dimension is equal
    OP_REQUIRES(
        context, input_tensor.dim_size(2) == filter_tensor.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();
    int batch_size = input_tensor.dim_size(1);
    cudaStream_t stream;
    int dim_timestep = input_tensor.dim_size(0);
    int dim_feature = input_tensor.dim_size(2);
    int dim_tau = filter_tensor.dim_size(0);
    cudaStreamCreate(&stream);
    dim3 grid(dim_timestep, batch_size);
    kernel<T><<<grid, dim_feature, 0, stream>>>(dim_tau, &input(0, 0, 0), &filter(0, 0), &output(0, 0, 0));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
};

REGISTER_KERNEL_BUILDER(Name("Lookahead").Device(DEVICE_GPU), LookaheadOp<float, 1>);
REGISTER_KERNEL_BUILDER(Name("Lookaheadgpu").Device(DEVICE_GPU), LookaheadOp<float, 1>);

