#include "Driver/GPU/HipApi.h"
#include "Driver/Dispatch.h"
#include "hip/hip_runtime_api.h"
#include <string>

namespace proton {

namespace hip {

struct ExternLibHip : public ExternLibBase {
  using RetType = hipError_t;
  static constexpr const char *name = "libamdhip64.so";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = hipSuccess;
  static void *lib;
};

void *ExternLibHip::lib = nullptr;

DEFINE_DISPATCH(ExternLibHip, deviceSynchronize, hipDeviceSynchronize)

DEFINE_DISPATCH(ExternLibHip, deviceGetAttribute, hipDeviceGetAttribute, int *,
                hipDeviceAttribute_t, int);

DEFINE_DISPATCH(ExternLibHip, getDeviceCount, hipGetDeviceCount, int *);

DEFINE_DISPATCH(ExternLibHip, getDeviceProperties, hipGetDeviceProperties,
                hipDeviceProp_t *, int);

Device getDevice(uint64_t index) {
  int clockRate;
  (void)hip::deviceGetAttribute<true>(&clockRate, hipDeviceAttributeClockRate,
                                      index);
  int memoryClockRate;
  (void)hip::deviceGetAttribute<true>(&memoryClockRate,
                                      hipDeviceAttributeMemoryClockRate, index);
  int busWidth;
  (void)hip::deviceGetAttribute<true>(&busWidth,
                                      hipDeviceAttributeMemoryBusWidth, index);
  int smCount;
  (void)hip::deviceGetAttribute<true>(
      &smCount, hipDeviceAttributeMultiprocessorCount, index);

  std::string arch = getHipArchName(index);

  return Device(DeviceType::HIP, index, clockRate, memoryClockRate, busWidth,
                smCount, arch);
}

// TODO: hipDeviceProp_t was updated to point from hipDeviceProp_tR0000 ->
// hipDeviceProp_tR0600 as part of a breaking API change in Rocm 6.0
// https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/driver.c
// uses hipDeviceProp_tR0000 and imports the hip_deprecated.h header file to be
// be back compatible with ROCm 5.x. PyTorch stills needs to support 5.x and the
// hipDeviceProp_tR0600 symbol does not exist pre-Rocm 6.0. Calling
// hipDeviceProp_tR0000 here with Rocm 6.1 causes a stack corruption. Therefore
// were will use hipDeviceProp_t and investigate if we can unify the definitions
// in the two files.

const std::string getHipArchName(uint64_t index) {
  hipDeviceProp_t devProp;
  (void)hip::getDeviceProperties<true>(&devProp, index);
  std::string gcnArchName(devProp.gcnArchName);
  std::string hipArch = gcnArchName.substr(0, 6);
  return hipArch;
}

const char *getKernelNameRef(const hipFunction_t f) {
  typedef const char *(*hipKernelNameRef_t)(const hipFunction_t);
  static hipKernelNameRef_t func = nullptr;
  Dispatch<ExternLibHip>::init(ExternLibHip::name, &ExternLibHip::lib);
  if (func == nullptr)
    func = reinterpret_cast<hipKernelNameRef_t>(
        dlsym(ExternLibHip::lib, "hipKernelNameRef"));
  return (func ? func(f) : NULL);
}

const char *getKernelNameRefByPtr(const void *hostFunction,
                                  hipStream_t stream) {
  typedef const char *(*hipKernelNameRefByPtr_t)(const void *, hipStream_t);
  static hipKernelNameRefByPtr_t func = nullptr;
  Dispatch<ExternLibHip>::init(ExternLibHip::name, &ExternLibHip::lib);
  if (func == nullptr)
    func = reinterpret_cast<hipKernelNameRefByPtr_t>(
        dlsym(ExternLibHip::lib, "hipKernelNameRefByPtr"));
  return (func ? func(hostFunction, stream) : NULL);
}

} // namespace hip

} // namespace proton
