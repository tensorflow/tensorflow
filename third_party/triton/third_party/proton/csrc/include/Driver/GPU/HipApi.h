#ifndef PROTON_DRIVER_GPU_HIP_H_
#define PROTON_DRIVER_GPU_HIP_H_

#include "Driver/Device.h"
#include "hip/hip_runtime_api.h"

namespace proton {

namespace hip {

template <bool CheckSuccess> hipError_t deviceSynchronize();

template <bool CheckSuccess>
hipError_t deviceGetAttribute(int *value, hipDeviceAttribute_t attribute,
                              int deviceId);

template <bool CheckSuccess> hipError_t getDeviceCount(int *count);

template <bool CheckSuccess>
hipError_t getDeviceProperties(hipDeviceProp_t *prop, int deviceId);

Device getDevice(uint64_t index);

const std::string getHipArchName(uint64_t index);

const char *getKernelNameRef(const hipFunction_t f);
const char *getKernelNameRefByPtr(const void *hostFunction, hipStream_t stream);

} // namespace hip

} // namespace proton

#endif // PROTON_DRIVER_GPU_HIP_H_
