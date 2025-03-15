// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_SCHEMA_SOC_TABLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_SCHEMA_SOC_TABLE_H_

#include <string>
#include <vector>

namespace qnn {
enum class SnapdragonModel {
  UNKNOWN_SDM = 0,
  SA8295 = 39,
  SM8350 = 30,
  SM8450 = 36,
  SM8550 = 43,
  SA8255 = 52,
  SM8650 = 57,
  SM8750 = 69
};

enum class DspArch {
  NONE = 0,
  V65 = 65,
  V66 = 66,
  V68 = 68,  // HTP supported device
  V69 = 69,  // HTP supported device
  V73 = 73,  // HTP supported device
  V75 = 75,  // HTP supported device
  V79 = 79,  // HTP supported device
};

struct SocInfo {
  const char* soc_name;
  SnapdragonModel soc_model;
  DspArch dsp_arch;
  std::size_t vtcm_size_in_mb;

  constexpr SocInfo(const char* soc_name, const SnapdragonModel soc_model,
                    const DspArch dsp_arch, const std::size_t vtcm_size_in_mb)
      : soc_name(soc_name),
        soc_model(soc_model),
        dsp_arch(dsp_arch),
        vtcm_size_in_mb(vtcm_size_in_mb) {}
};

constexpr SocInfo kSocInfos[] = {
    {SocInfo("UNKNOWN_SDM", SnapdragonModel::UNKNOWN_SDM, DspArch::NONE,
             0  // vtcm_size_in_mb
             )},
    {SocInfo("SA8255", SnapdragonModel::SA8255, DspArch::V73,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SA8295", SnapdragonModel::SA8295, DspArch::V68,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8350", SnapdragonModel::SM8350, DspArch::V68,
             4  // vtcm_size_in_mb
             )},
    {SocInfo("SM8450", SnapdragonModel::SM8450, DspArch::V69,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8550", SnapdragonModel::SM8550, DspArch::V73,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8650", SnapdragonModel::SM8650, DspArch::V75,
             8  // vtcm_size_in_mb
             )},
    {SocInfo("SM8750", SnapdragonModel::SM8750, DspArch::V79,
             8  // vtcm_size_in_mb
             )},
};
}  // namespace qnn
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_SCHEMA_SOC_TABLE_H_
