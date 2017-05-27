#if GOOGLE_CUDA
#define EIGEN_USE_GPU
__global__ void RoiPoolingGpuKernel(   const float* data_array, 
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
    int N = batch_size * num_channels * num_rois;
  
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x) {
        
        int batch, channel, roi_index;
        
        roi_index = i % num_rois;
        channel = ((i - roi_index) % (num_rois * num_channels)) / (num_rois);
        batch = (i - roi_index - channel * num_rois) / (num_rois * num_channels);
            
        // Calculate image index
        int image_index = channel + batch * num_channels;
        
        // Get the offset of the start of this image
        int image_start = image_index * image_width * image_height;
                
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
        // TODO more robust integer division
        // TODO handle size 0 case
        int kernel_width = roi_width / pooling_width;
        int kernel_height = roi_height / pooling_height;
        
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
                        
                        // If the value is bigger than we've seen, replace
                        if (value_at_index > current_max) {
                            current_max = value_at_index;
                            //LOG(INFO) << pixel;
                            argmax = pixel;
                        }
                    }
                }
                
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

__global__ void RoiPoolingGradGpuKernel(const int* argmax_array, const float* grad_array, float* output_array, int input_length) {
    int N = input_length;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x) {
        
        // Get the location to put this value
        int location = argmax_array[i];
        
        // make sure the value is valid
        if (location >= 0) {
            // Put the grad value in the output
            atomicAdd(output_array + location, grad_array[i]);
        }
    }
}

__global__ void ZeroTensorKernel(float* input, int length) {
    int N = length;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x) {
        
        input[i] = 0;
    }
}

// *****************************************************
// Kernel launchers
// *****************************************************

void RoiPoolingGpuKernelLauncher(      const float* data_array, 
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
    
    RoiPoolingGpuKernel<<<32, 256>>>(  data_array, 
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

void RoiPoolingGradGpuKernelLauncher(const int* argmax_array, const float* grad_array, float* output_array, int input_length) {
    RoiPoolingGradGpuKernel<<<32, 256>>>(  argmax_array, grad_array, output_array, input_length);
}

void ZeroTensorGpuKernelLauncher(float* input, int length) {
    ZeroTensorKernel<<<32, 256>>>(input, length);
}

#endif
