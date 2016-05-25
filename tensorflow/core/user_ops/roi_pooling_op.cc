#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

REGISTER_OP("RoiPooling")
    .Input("conv_features: float")
    .Input("rois: int32")
    .Input("output_size: int32")
    .Output("output_tensor: float")
    .Output("argmax: int32");
   
// Computes the forward pass of the ROI pooling operation
class RoiPoolingOp : public OpKernel {
    public:
    explicit RoiPoolingOp(OpKernelConstruction* context) : OpKernel(context) { }

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
        
        // Iterate through all batches
        for (int batch = 0; batch < batch_size; ++batch) {
            
            // Iterate through all channel
            for (int channel = 0; channel < num_channels; ++channel) {
                
                // Calculate image index
                int image_index = channel + batch * num_channels;
                
                // Get the offset of the start of this image
                int image_start = image_index * image_width * image_height;
                
                // Get current ROI
                for (int roi_index = 0; roi_index < num_rois; ++roi_index) {
                    
                    // Get the current ROI
                    int* current_roi = roi_data + (batch * num_rois * 4) + roi_index * 4;
                    
                    // Get dimensions of current ROI
                    int roi_x = current_roi[0];
                    int roi_y = current_roi[1];
                    int roi_width = current_roi[3];
                    int roi_height = current_roi[2];
                    
                    // Get offset into frame of first pixel of ROI
                    int roi_offset = roi_x + roi_y * image_width;
                    
                    // Get the offset of the start of this ROI
                    int roi_start = image_start + roi_offset;
                    
                    // Determine pooling kernel size
                    // TODO handle cleaner integer division
                    // TODO handle size 0 case
                    int kernel_width = roi_width / pooling_width;
                    int kernel_height = roi_height / pooling_height;
                    
                    LOG(INFO) << num_rois << " rois of size ROI Dims: " << roi_width << "by " << roi_height;
                    LOG(INFO) << "Pooling Dims: " << pooling_width << "by " << pooling_height;
                    
                    // Iterate through each pooling region
                    for (int rx = 0; rx < pooling_width; ++rx) {
                        for (int ry = 0; ry < pooling_height; ++ry) {
                            
                            // Get the offset of the start of this pooling region
                            int region_start = roi_start + (rx * kernel_width) + (ry * kernel_height) * image_width;
                            
                            float current_max = 0;
                            int argmax = -1;
                            
                            //LOG(INFO) << "Kernel Dims: " << kernel_width << "by " << kernel_height;
                            
                            // Iterate through each pixel in the pooling region
                            for (int px = 0; px < kernel_width; ++px) {
                                for (int py = 0; py < kernel_height; ++py) {

                                    // Get location of pixel in image
                                    int pixel = region_start + px + py * image_width;
                                    
                                    // Get value at this location
                                    float value_at_index = data_array[pixel];
                                    
                                    //LOG(INFO) << "Pixel Value" << value_at_index;
                                    
                                    // If the value is bigger than we've seen, replace
                                    if (value_at_index > current_max) {
                                        current_max = value_at_index;
                                        //LOG(INFO) << pixel;
                                        argmax = pixel;
                                    }
                                }
                            }
                            
                            //LOG(INFO) << "Current Max" << current_max;
                            
                            // Save max value for pooling region to output
                            // Output array will have shape (batch_id, channel_id, roi, x, y)
                            int output_size = pooling_width * pooling_height;
                            output_array[image_index * output_size * num_rois 
                                + roi_index * output_size + rx + ry * pooling_width] = current_max;
                            
                            // Save location of max value in argmax array
                            argmax_array[image_index * output_size * num_rois 
                                + roi_index * output_size + rx + ry * pooling_width] = argmax;
                            
                        }
                    }
                }
            }   
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("RoiPooling").Device(DEVICE_CPU), RoiPoolingOp);


REGISTER_OP("RoiPoolingGrad")
    .Input("grad: float")
    .Input("argmax: int32")
    .Input("prop_grad_shape: int32")
    .Output("prop_grad: float");
    
// Computes the gradent of the ROI pooling layer
class RoiPoolingGradOp : public OpKernel {
    public:
    explicit RoiPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) { }

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
        
        // Go through all elements in grad/argmax and put the grad value in the location 
        // determined by argmax
        for (int i = 0; i < input_length; ++i) {
            
            // Get the location to put this value
            int location = argmax_array[i];
            LOG(INFO) << "i: " << i << " location : " << location;
            
            // make sure the value is valid
            if (location >= 0) {
                LOG(INFO) << "writing: " << grad_array[i];
                // Put the grad value in the output
                output_array[location] = grad_array[i];
            }
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad").Device(DEVICE_CPU), RoiPoolingGradOp);

