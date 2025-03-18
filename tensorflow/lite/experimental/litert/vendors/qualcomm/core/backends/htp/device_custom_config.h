// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_DEVICE_CUSTOM_CONFIG_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_DEVICE_CUSTOM_CONFIG_H_

#include <memory>
#include <vector>

#include "third_party/qairt/latest/include/QNN/HTP/QnnHtpDevice.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/schema/soc_table.h"
namespace qnn {
class HtpDeviceCustomConfig {
 public:
  std::vector<QnnDevice_CustomConfig_t> CreateDeviceCustomConfig(
      const SocInfo* soc_info);

  QnnHtpDevice_CustomConfig_t* AllocDeviceCustomConfig() {
    htp_device_config_.emplace_back(
        std::make_unique<QnnHtpDevice_CustomConfig_t>());
    htp_device_config_.back()->option = QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN;
    return htp_device_config_.back().get();
  }

 private:
  std::vector<std::unique_ptr<QnnHtpDevice_CustomConfig_t>> htp_device_config_;
};
}  // namespace qnn
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_BACKENDS_HTP_DEVICE_CUSTOM_CONFIG_H_
