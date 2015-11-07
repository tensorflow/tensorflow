#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

void GPUDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  GPUUtil::CopyCPUTensorToGPU(cpu_tensor, this, device, device_tensor, done);
}

void GPUDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             const string& tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  GPUUtil::CopyGPUTensorToCPU(device, this, device_tensor, cpu_tensor, done);
}

}  // namespace tensorflow
