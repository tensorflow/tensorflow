// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/schema/soc_table.h"
namespace qnn {
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
constexpr unsigned long kNumSocInfos =
    sizeof(::qnn::kSocInfos) / sizeof(::qnn::kSocInfos[0]);
}  // namespace qnn
