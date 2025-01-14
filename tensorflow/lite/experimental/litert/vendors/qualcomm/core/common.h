// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum LiteRtQnnLogLevel {  // NOLINT(modernize-use-using)
  /// Disable delegate and QNN backend logging messages.
  kLogOff = 0,
  kLogLevelError = 1,
  kLogLevelWarn = 2,
  kLogLevelInfo = 3,
  kLogLevelVerbose = 4,
  kLogLevelDebug = 5,
} LiteRtQnnLogLevel;

typedef enum LiteRtQnnHtpPerformanceMode {  // NOLINT(modernize-use-using)
  kHtpDefault = 0,
  kHtpSustainedHighPerformance = 1,
  kHtpBurst = 2,
  kHtpHighPerformance = 3,
  kHtpPowerSaver = 4,
  kHtpLowPowerSaver = 5,
  kHtpHighPowerSaver = 6,
  kHtpLowBalanced = 7,
  kHtpBalanced = 8,
  kHtpExtremePowerSaver = 9,
} LiteRtQnnHtpPerformanceMode;

typedef struct {  // NOLINT
  /// The default performance mode sets no configurations on the HTP.
  LiteRtQnnHtpPerformanceMode performance_mode;
} LiteRtQnnHtpBackendOptions;

// This option can be used to specify QNN options.
static const char* kDispatchOptionLiteRtQnnOptions = "litert_qnn_options";

typedef struct {  // NOLINT
  /// Optional backend specific options for the HTP backend.
  LiteRtQnnHtpBackendOptions htp_options;
  /// Log level
  LiteRtQnnLogLevel log_level;
} LiteRtQnnOptions;

#define LITERT_QNN_HTP_OPTION_INIT \
  { kHtpDefault /*performance_mode*/ }
#define LITERT_QNN_OPTION_INIT                  \
  {                                             \
    LITERT_QNN_HTP_OPTION_INIT, /*htp_options*/ \
        kLogOff,                /*log_level*/   \
  }

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
