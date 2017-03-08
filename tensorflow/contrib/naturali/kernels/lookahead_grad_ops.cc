#include "tensorflow/contrib/naturali/kernels/lookahead_grad_ops.h"

using namespace tensorflow;

template<typename T>
class LookaheadGradOp<T, 0> : public OpKernel {
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

    for (int timestep = 0; timestep < input_tensor.dim_size(0); timestep++) {
      for (int batch = 0; batch < input_tensor.dim_size(1); batch++) {
        for (int feature = 0; feature < input_tensor.dim_size(2); feature++) {
          output(timestep, batch, feature) = 0;
          for (int input_begin = 0; input_begin < filter_tensor.dim_size(0); input_begin++) {
            int index = input_begin + timestep - filter_tensor.dim_size(0) + 1;
            int filter_idx = filter_tensor.dim_size(0) - 1 - input_begin;
            if (index >= 0 && filter_idx >= 0) {
              output(timestep, batch, feature) += output_grad(index, batch, feature) * filter(filter_idx, feature);
            }
          }
        }
      }
    }
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();

    for (int tau = 0; tau < filter_tensor.dim_size(0); tau++) {
      for (int feature = 0; feature < filter_tensor.dim_size(1); feature++) {
        output2(tau, feature) = 0;
      }
    }
    for (int batch = 0; batch < input_tensor.dim_size(1); batch++) {
      for (int feature = 0; feature < filter_tensor.dim_size(1); feature++) {
        for (int tau = 0; tau < filter_tensor.dim_size(0); tau++) {
          for (int timestep = 0; timestep < input_tensor.dim_size(0) - tau; timestep++) {
            output2(tau, feature) += output_grad(timestep, batch, feature) * input(timestep + tau, batch, feature);
          }
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgrad").Device(DEVICE_CPU), LookaheadGradOp<float, 0>);

