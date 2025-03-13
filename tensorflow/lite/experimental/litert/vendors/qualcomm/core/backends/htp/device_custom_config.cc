// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/backends/htp/device_custom_config.h"

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/schema/soc_table.h"
namespace qnn {
std::vector<QnnDevice_CustomConfig_t>
HtpDeviceCustomConfig::CreateDeviceCustomConfig(const SocInfo* soc_info) {
  std::vector<QnnDevice_CustomConfig_t> ret;
  QnnHtpDevice_CustomConfig_t* p_custom_config = nullptr;

  p_custom_config = AllocDeviceCustomConfig();
  p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  p_custom_config->socModel = static_cast<uint32_t>(soc_info->soc_model);
  ret.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));

  return ret;
}
}  // namespace qnn
