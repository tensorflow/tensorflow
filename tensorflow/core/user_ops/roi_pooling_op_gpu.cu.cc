

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void RoiPoolingKernel(   const float* data_array, 
                                    float* output_array, 
                                    int* argmax_array, 
                                    const int* roi_data,
                                    int batch_size,
                                    int num_channels,
                                    int image_width,
                                    int image_height, 
                                    int pooling_width,
                                    int pooling_height,
                                    int num_rois) {
    int N = batch_size;
  
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x) {
        
        int batch = i;
        
        // Iterate through all channel
        for (int channel = 0; channel < num_channels; ++channel) {
            
            // Calculate image index
            int image_index = channel + batch * num_channels;
            
            // Get the offset of the start of this image
            int image_start = image_index * image_width * image_height;
            
            // Get current ROI
            for (int roi_index = 0; roi_index < num_rois; ++roi_index) {
                
                // Get the current ROI
                const int* current_roi = roi_data + (batch * num_rois * 4) + roi_index * 4;
                
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
                
                //LOG(INFO) << num_rois << " rois of size ROI Dims: " << roi_width << "by " << roi_height;
                //LOG(INFO) << "Pooling Dims: " << pooling_width << "by " << pooling_height;
                
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
                                    int num_rois) {
    
    // TODO get rid of magic numbers
    
    RoiPoolingKernel<<<32, 256>>>(  data_array, 
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




__global__ void RoiPoolingGradKernel(const int* argmax_array, const float* grad_array, float* output_array, int input_length) {
    int N = input_length;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x) {
        
        // Get the location to put this value
        int location = argmax_array[i];
        //LOG(INFO) << "i: " << i << " location : " << location;
        
        //output_array[location] = .25;
        
        // make sure the value is valid
        if (location >= 0) {
            //LOG(INFO) << "writing: " << grad_array[i];
            // Put the grad value in the output
            output_array[location] = grad_array[i];
            //output_array[location] = .25;
        }
    }
}

void RoiPoolingGradKernelLauncher(const int* argmax_array, const float* grad_array, float* output_array, int input_length) {
    
    // TODO get rid of magic numbers
    
    RoiPoolingGradKernel<<<32, 256>>>(  argmax_array, grad_array, output_array, input_length);
}




__global__ void ZeroTensorKernel(float* input, int length) {
    int N = length;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x) {
        
        input[i] = 0;
    }
}

void ZeroTensorKernelLauncher(float* input, int length) {
    
    // TODO get rid of magic numbers
    
    ZeroTensorKernel<<<32, 256>>>(input, length);
}



#endif
