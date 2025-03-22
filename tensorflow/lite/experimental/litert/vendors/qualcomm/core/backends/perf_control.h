// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_PERF_CONTROL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_PERF_CONTROL_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/common.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"

namespace qnn {
template <typename T>
std::vector<std::add_pointer_t<std::add_const_t<T>>> ObtainNullTermPtrVector(
    const std::vector<T>& vec) {
  std::vector<std::add_pointer_t<std::add_const_t<T>>> ret(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    ret[i] = &(vec[i]);
  }
  ret.push_back(nullptr);
  return ret;
}

// Defines Qnn performance mode vote types for htpbackend
enum PerformanceModeVoteType {
  kNoVote = 0,
  kUpVote = 1,
  kDownVote = 2,
};
class PerfControl {
 public:
  explicit PerfControl(const QNN_INTERFACE_VER_TYPE* api,
                       const HtpBackendOptions& htp_options);
  PerfControl(const PerfControl&) = delete;
  PerfControl(PerfControl&&) = delete;
  PerfControl& operator=(const PerfControl&) = delete;
  PerfControl& operator=(PerfControl&&) = delete;
  ~PerfControl();

  bool Init(Qnn_DeviceHandle_t* device_handle);
  bool Terminate();
  // Direct vote is only supported in manual mode.
  void PerformanceVote();
  void ReleasePerformanceVote();
  bool CreatePerfPowerConfigPtr(const std::uint32_t power_config_id,
                                const HtpPerformanceMode perf_mode,
                                const PerformanceModeVoteType vote_type);

 private:
  inline bool IsPerfModeEnabled() const {
    return performance_mode_ != HtpPerformanceMode::kHtpDefault;
  }
  const QNN_INTERFACE_VER_TYPE* api_{nullptr};
  struct BackendConfig;
  std::unique_ptr<BackendConfig> backend_config_;
  std::uint32_t powerconfig_client_id_{0};
  PerformanceModeVoteType manual_voting_type_{kNoVote};
  // HTPBackendOptions
  HtpPerformanceMode performance_mode_{kHtpDefault};
  std::uint32_t device_id_{0};
};
}  // namespace qnn
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_PERF_CONTROL_H_
