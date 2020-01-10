#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"

namespace perftools {
namespace gputools {
class Stream;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

namespace gpu = ::perftools::gputools;

class GPUDeviceContext : public DeviceContext {
 public:
  GPUDeviceContext(int stream_id, gpu::Stream* stream)
      : stream_id_(stream_id), stream_(stream) {}

  ~GPUDeviceContext() override {}

  gpu::Stream* stream() const override { return stream_; }
  int stream_id() const { return stream_id_; }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             const string& edge_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override;

  void MaintainLifetimeOnStream(
      const Tensor* t, perftools::gputools::Stream* stream) const override {}

 private:
  int stream_id_;
  gpu::Stream* stream_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
