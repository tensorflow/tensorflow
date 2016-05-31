#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

void RoiPoolingKernelLauncher(      const float* data_array, 
                                    float* output_array, 
                                    int* argmax_array, 
                                    const int* roi_data,
                                    int batch_size,
                                    int num_channels,
                                    int image_width,
                                    int image_height, 
                                    int pooling_width,
                                    int pooling_height,
                                    int num_rois);



REGISTER_OP("RoiPoolingGPU")
    .Input("conv_features: float")
    .Input("rois: int32")
    .Input("output_size: int32")
    .Output("output_tensor: float")
    .Output("argmax: int32");
   
// Computes the forward pass of the ROI pooling operation
class RoiPoolingOpGPU : public OpKernel {
    public:
    explicit RoiPoolingOpGPU(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override {
        
        
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& roi_tensor = context->input(1);
        
        // TODO check inputs properly
        // ROI same size as features, etc
        
        // Make sure the input tensor is 4d
        OP_REQUIRES(context, input_tensor.dims() == 4, 
                        errors::InvalidArgument("This op expects a 4d input tensor"));

        
        // ROIs should be passed in batch, ROI index order
        // ROIs for same image in a batch are the same - so no "channel"
        // Then x, y, height, width
        int num_rois = roi_tensor.dim_size(1);;
        
        // Get the input dimensions
        int batch_size = input_tensor.dim_size(0);
        int num_channels = input_tensor.dim_size(1);
        int image_width = input_tensor.dim_size(3);
        int image_height = input_tensor.dim_size(2);
        
        // Get the size of the output "image"
        // Should be format HW
        const Tensor& output_size_tensor = context->input(2);
        int* output_size_pointer = (int*) output_size_tensor.tensor_data().data();
        int pooling_width = output_size_pointer[1];
        int pooling_height = output_size_pointer[0];
        
        
        // Create the output feature tensor
        Tensor* output_tensor = NULL;
        TensorShape output_shape = {batch_size, num_channels, num_rois, pooling_height, pooling_width};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        
        // Create the argmax output tensor
        Tensor* argmax_tensor = NULL;
        TensorShape argmax_tensor_shape = {batch_size, num_channels, num_rois, pooling_height, pooling_width};
        OP_REQUIRES_OK(context, context->allocate_output(1, argmax_tensor_shape, &argmax_tensor));

        // Get the data arrays (a version we can access by index quickly)
        float* data_array = (float*) input_tensor.tensor_data().data();
        float* output_array = (float*) output_tensor->tensor_data().data();
        int* argmax_array = (int*) argmax_tensor->tensor_data().data();
        int* roi_data = (int*) roi_tensor.tensor_data().data();
        
        LOG(INFO) << "Calling CUDA kernel!";
        
        RoiPoolingKernelLauncher(   data_array, 
                                    output_array, 
                                    argmax_array, 
                                    roi_data,
                                    batch_size,
                                    num_channels,
                                    image_width,
                                    image_height, 
                                    pooling_width,
                                    pooling_height,
                                    num_rois);
       
    }
};


REGISTER_KERNEL_BUILDER(Name("RoiPoolingGPU")
                        .Device(DEVICE_GPU)
                        .HostMemory("output_size"), 
                        RoiPoolingOpGPU);

void RoiPoolingGradKernelLauncher(const int* argmax_array, const float* grad_array, float* output_array, int input_length);
void ZeroTensorKernelLauncher(float* input, int length);

REGISTER_OP("RoiPoolingGradGPU")
    .Input("grad: float")
    .Input("argmax: int32")
    .Input("prop_grad_shape: int32")
    .Output("prop_grad: float");
    
// Computes the gradent of the ROI pooling layer
class RoiPoolingGradGPUOp : public OpKernel {
    public:
    explicit RoiPoolingGradGPUOp(OpKernelConstruction* context) : OpKernel(context) { }

    void Compute(OpKernelContext* context) override {
           
        // TODO: Check inputs
        // Make sure argmax and grad have same shape, etc
        
        // Get the input gradients
        const Tensor& grad = context->input(0);
        float* grad_array = (float*) grad.tensor_data().data();
        
        // Get the input argmax data
        const Tensor& argmax = context->input(1);
        int* argmax_array = (int*) argmax.tensor_data().data();
        
        // Get the output shape
        const Tensor& shape = context->input(2);
        int* shape_array = (int*) shape.tensor_data().data();
        
        // Create the output grad tensor
        Tensor* output_tensor = NULL;
        TensorShape output_shape = {shape_array[0], shape_array[1], shape_array[2], shape_array[3]};
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        float* output_array = (float*) output_tensor->tensor_data().data();
        
        // The number of elements in grad and argmax
        int input_length = grad.dim_size(0) * grad.dim_size(1) * grad.dim_size(2) * grad.dim_size(3) * grad.dim_size(4);
        LOG(INFO) << "H: " << grad.dim_size(0) << " C: " << grad.dim_size(1) << " N: " << grad.dim_size(2) << " H: " << grad.dim_size(3) << " W: " << grad.dim_size(4);
        LOG(INFO) << "input length: " << input_length;
        
        // TODO record timings here and print
        ZeroTensorKernelLauncher(output_array, input_length);
        RoiPoolingGradKernelLauncher(argmax_array, grad_array, output_array, input_length);
    }
};


REGISTER_KERNEL_BUILDER(Name("RoiPoolingGradGPU").Device(DEVICE_GPU).HostMemory("prop_grad_shape"), RoiPoolingGradGPUOp);

