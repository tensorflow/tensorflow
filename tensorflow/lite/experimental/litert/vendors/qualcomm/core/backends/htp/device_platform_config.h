
// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_DEVICE_PLATFORM_CONFIG_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_DEVICE_PLATFORM_CONFIG_H_

#include <memory>
#include <vector>

#include "third_party/qairt/latest/include/QNN/HTP/QnnHtpDevice.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/schema/soc_table.h"

namespace qnn {
class HtpDevicePlatformInfoConfig {
 public:
  std::vector<QnnDevice_PlatformInfo_t*> CreateDevicePlatformInfo(
      const SocInfo* qcom_target_soc_info);

 private:
  QnnDevice_PlatformInfo_t* AllocDevicePlatformInfo() {
    htp_platform_info_.emplace_back(
        std::make_unique<QnnDevice_PlatformInfo_t>());
    htp_platform_info_.back()->version =
        QNN_DEVICE_PLATFORM_INFO_VERSION_UNDEFINED;
    return htp_platform_info_.back().get();
  }

  QnnDevice_HardwareDeviceInfo_t* AllocHwDeviceInfo() {
    htp_hw_device_info_.emplace_back(
        std::make_unique<QnnDevice_HardwareDeviceInfo_t>());
    htp_hw_device_info_.back()->version =
        QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_UNDEFINED;
    return htp_hw_device_info_.back().get();
  }

  QnnDevice_CoreInfo_t* AllocCoreInfo() {
    htp_core_info_.emplace_back(std::make_unique<QnnDevice_CoreInfo_t>());
    htp_core_info_.back()->version = QNN_DEVICE_CORE_INFO_VERSION_UNDEFINED;
    return htp_core_info_.back().get();
  }

  QnnHtpDevice_DeviceInfoExtension_t* AllocDeviceInfoExtension() {
    htp_device_info_extension_.emplace_back(
        std::make_unique<QnnHtpDevice_DeviceInfoExtension_t>());
    htp_device_info_extension_.back()->devType = QNN_HTP_DEVICE_TYPE_UNKNOWN;
    return htp_device_info_extension_.back().get();
  }

  std::vector<std::unique_ptr<QnnDevice_PlatformInfo_t>> htp_platform_info_;
  std::vector<std::unique_ptr<QnnDevice_HardwareDeviceInfo_t>>
      htp_hw_device_info_;
  std::vector<std::unique_ptr<QnnDevice_CoreInfo_t>> htp_core_info_;
  std::vector<std::unique_ptr<QnnHtpDevice_DeviceInfoExtension_t>>
      htp_device_info_extension_;
};
}  // namespace qnn
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_DEVICE_PLATFORM_CONFIG_H_
