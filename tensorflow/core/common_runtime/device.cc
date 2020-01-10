#include "tensorflow/core/common_runtime/device.h"

#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

Device::Device(Env* env, const DeviceAttributes& device_attributes,
               Allocator* device_allocator)
    : DeviceBase(env), device_attributes_(device_attributes) {
  CHECK(DeviceNameUtils::ParseFullName(name(), &parsed_name_))
      << "Invalid device name: " << name();
  rmgr_ = new ResourceMgr(parsed_name_.job);
}

Device::~Device() { delete rmgr_; }

// static
DeviceAttributes Device::BuildDeviceAttributes(
    const string& name, DeviceType device, Bytes memory_limit,
    BusAdjacency bus_adjacency, const string& physical_device_desc) {
  DeviceAttributes da;
  da.set_name(name);
  do {
    da.set_incarnation(random::New64());
  } while (da.incarnation() == 0);  // This proto field must not be zero
  da.set_device_type(device.type());
  da.set_memory_limit(memory_limit.value());
  da.set_bus_adjacency(bus_adjacency);
  da.set_physical_device_desc(physical_device_desc);
  return da;
}

}  // namespace tensorflow
