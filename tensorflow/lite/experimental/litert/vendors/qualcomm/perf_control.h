// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_PERF_CONTROL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_PERF_CONTROL_H_

#include <memory>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"

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
  explicit PerfControl(const QnnApi* api,
                      const LiteRtQnnHtpBackendOptions& htp_options);
  PerfControl(const PerfControl&) = delete;
  PerfControl(PerfControl&&) = delete;
  PerfControl& operator=(const PerfControl&) = delete;
  PerfControl& operator=(PerfControl&&) = delete;
  ~PerfControl();

  LiteRtStatus Init(Qnn_DeviceHandle_t* device_handle);
  LiteRtStatus Terminate();
  // Direct vote is only supported in manual mode.
  void PerformanceVote();
  void ReleasePerformanceVote();
  LiteRtStatus CreatePerfPowerConfigPtr(
      const std::uint32_t power_config_id,
      const LiteRtQnnHtpPerformanceMode perf_mode,
      const PerformanceModeVoteType vote_type);

 private:
  inline bool IsPerfModeEnabled() const {
    return performance_mode_ !=
           LiteRtQnnHtpPerformanceMode::kHtpDefault;
  }
  const QnnApi* api_{nullptr};
  struct BackendConfig;
  std::unique_ptr<BackendConfig> backend_config_;
  std::uint32_t powerconfig_client_id_{0};
  PerformanceModeVoteType manual_voting_type_{kNoVote};
  // HTPBackendOptions
  LiteRtQnnHtpPerformanceMode performance_mode_{kHtpDefault};
  std::uint32_t device_id_{0};
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_PERF_CONTROL_H_
