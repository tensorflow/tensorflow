#ifndef TENSORFLOW_COMMON_RUNTIME_LOCAL_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_LOCAL_DEVICE_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {

class SessionOptions;

// This class is shared by ThreadPoolDevice and GPUDevice and
// initializes a shared Eigen compute device used by both.  This
// should eventually be removed once we refactor ThreadPoolDevice and
// GPUDevice into more 'process-wide' abstractions.
class LocalDevice : public Device {
 public:
  LocalDevice(const SessionOptions& options, const DeviceAttributes& attributes,
              Allocator* device_allocator);
  ~LocalDevice() override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LocalDevice);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_LOCAL_DEVICE_H_
