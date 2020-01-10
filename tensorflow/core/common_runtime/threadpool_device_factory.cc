// Register a factory that provides CPU devices.
#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// TODO(zhifengc/tucker): Figure out the bytes of available RAM.
class ThreadPoolDeviceFactory : public DeviceFactory {
 public:
  void CreateDevices(const SessionOptions& options, const string& name_prefix,
                     std::vector<Device*>* devices) override {
    // TODO(zhifengc/tucker): Figure out the number of available CPUs
    // and/or NUMA configuration.
    int n = 1;
    auto iter = options.config.device_count().find("CPU");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/cpu:", i);
      devices->push_back(new ThreadPoolDevice(options, name, Bytes(256 << 20),
                                              BUS_ANY, cpu_allocator()));
    }
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("CPU", ThreadPoolDeviceFactory);

}  // namespace tensorflow
