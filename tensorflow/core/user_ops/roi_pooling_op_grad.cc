#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

// CUDA launcher function prototypes
void RoiPoolingGradGpuKernelLauncher(const int* argmax_array, const float* grad_array, float* output_array, int input_length);
void ZeroTensorGpuKernelLauncher(float* input, int length);

void RoiPoolingGradCpuKernel(const int* argmax_array, const float* grad_array, float* output_array, int input_length) {
    // Go through all elements in grad/argmax and put the grad value in the location 
    // determined by argmax
    for (int i = 0; i < input_length; ++i) {
        
        // Get the location to put this value
        int location = argmax_array[i];
        
        // make sure the value is valid
        if (location >= 0) {
            // Put the grad value in the output
            output_array[location] += grad_array[i];
        }
    }
}

void ComputeRoiPoolingGrad(OpKernelContext* context, bool useGPU) {
    
    // Get the input gradients
    const Tensor& grad = context->input(0);
    float* grad_array = (float*) grad.tensor_data().data();
    
    // Get the input argmax data
    const Tensor& argmax = context->input(1);
    int* argmax_array = (int*) argmax.tensor_data().data();
    
    // Get the output shape
    const Tensor& shape = context->input(2);
    int* shape_array = (int*) shape.tensor_data().data();
    
    // Check that the input tensors are valid
    OP_REQUIRES(context, grad.dims() == 5, 
                    errors::InvalidArgument("This op expects a 5d gradient tensor"));
    
    OP_REQUIRES(context, argmax.dims() == 5, 
                    errors::InvalidArgument("This op expects a 5d argmax tensor"));
    
    OP_REQUIRES(context, shape.dims() == 1, 
                    errors::InvalidArgument("This op expects a 1d shape tensor"));
    
    OP_REQUIRES(context, shape.dim_size(0) == 4, 
                    errors::InvalidArgument("Shape should have a first dimension of length 4"));
    
    OP_REQUIRES(context, grad.dim_size(0) == argmax.dim_size(0) &&
                           grad.dim_size(1) == argmax.dim_size(1) &&
                           grad.dim_size(2) == argmax.dim_size(2) &&
                           grad.dim_size(3) == argmax.dim_size(3) &&
                           grad.dim_size(4) == argmax.dim_size(4),
                    errors::InvalidArgument("Argmax and Grad tensors should be of same shape"));
    
    OP_REQUIRES(context, shape_array[0] == argmax.dim_size(0) &&
                           shape_array[1] == argmax.dim_size(1),
                    errors::InvalidArgument("Shape should match argmax/grad for first two dimensions"));
    
    // Create the output grad tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape = {shape_array[0], shape_array[1], shape_array[2], shape_array[3]};
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    float* output_array = (float*) output_tensor->tensor_data().data();
    
    // The number of elements in grad and argmax
    int input_length = grad.dim_size(0) * grad.dim_size(1) * grad.dim_size(2) * grad.dim_size(3) * grad.dim_size(4);
    int output_length = shape_array[0] * shape_array[1] * shape_array[2] * shape_array[3];
    
    if (useGPU) {
        // Set the output array to zero
        ZeroTensorGpuKernelLauncher(output_array, output_length);
        // run the kernel
        RoiPoolingGradGpuKernelLauncher(argmax_array, grad_array, output_array, input_length);
    }
    else {
        // Set the output array to zero
        memset(output_array, 0, output_length * sizeof(float));
        // Run the kernel
        RoiPoolingGradCpuKernel(argmax_array, grad_array, output_array, input_length);
    }
}

    
// Computes the gradent of the ROI pooling layer
class RoiPoolingGradCpu : public OpKernel {
    public:
    explicit RoiPoolingGradCpu(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override {
       // Compute using the CPU
       ComputeRoiPoolingGrad(context, false);
    }
};

// Computes the gradent of the ROI pooling ComputeROIlayer
class RoiPoolingGradGpu : public OpKernel {
    public:
    explicit RoiPoolingGradGpu(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override {
       // Compute using the GPU
       ComputeRoiPoolingGrad(context, true);
    }
};

REGISTER_OP("RoiPoolingGrad")
    .Input("grad: float")
    .Input("argmax: int32")
    .Input("prop_grad_shape: int32")
    .Output("prop_grad: float");

REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad").Device(DEVICE_CPU), RoiPoolingGradCpu);
REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad").Device(DEVICE_GPU).HostMemory("prop_grad_shape"), RoiPoolingGradGpu);
