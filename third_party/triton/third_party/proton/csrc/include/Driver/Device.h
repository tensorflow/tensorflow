#ifndef PROTON_DRIVER_DEVICE_H_
#define PROTON_DRIVER_DEVICE_H_

#include <cstdint>
#include <string>

namespace proton {

enum class DeviceType { HIP, CUDA, COUNT };

template <DeviceType T> struct DeviceTraits;

template <> struct DeviceTraits<DeviceType::CUDA> {
  constexpr static DeviceType type = DeviceType::CUDA;
  constexpr static const char *name = "CUDA";
};

template <> struct DeviceTraits<DeviceType::HIP> {
  constexpr static DeviceType type = DeviceType::HIP;
  constexpr static const char *name = "HIP";
};

struct Device {
  DeviceType type;
  uint64_t id;
  uint64_t clockRate;       // khz
  uint64_t memoryClockRate; // khz
  uint64_t busWidth;
  uint64_t numSms;
  std::string arch;

  Device() = default;

  Device(DeviceType type, uint64_t id, uint64_t clockRate,
         uint64_t memoryClockRate, uint64_t busWidth, uint64_t numSms,
         std::string arch)
      : type(type), id(id), clockRate(clockRate),
        memoryClockRate(memoryClockRate), busWidth(busWidth), numSms(numSms),
        arch(arch) {}
};

Device getDevice(DeviceType type, uint64_t index);

const std::string getDeviceTypeString(DeviceType type);

}; // namespace proton

#endif // PROTON_DRIVER_DEVICE_H_
