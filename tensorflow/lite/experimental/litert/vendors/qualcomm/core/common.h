// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum QnnLogLevel {  // NOLINT(modernize-use-using)
  /// Disable delegate and QNN backend logging messages.
  kLogOff = 0,
  kLogLevelError = 1,
  kLogLevelWarn = 2,
  kLogLevelInfo = 3,
  kLogLevelVerbose = 4,
  kLogLevelDebug = 5,
} QnnLogLevel;

typedef enum HtpPerformanceMode {  // NOLINT(modernize-use-using)
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
} HtpPerformanceMode;

typedef struct {  // NOLINT
  /// The default performance mode sets no configurations on the HTP.
  HtpPerformanceMode performance_mode;
} HtpBackendOptions;

typedef struct {  // NOLINT
  /// Optional backend specific options for the HTP backend.
  HtpPerformanceMode htp_options;
  /// Log level
  QnnLogLevel log_level;
} QnnOptions;

#define HTP_OPTION_INIT \
  { kHtpDefault /*performance_mode*/ }
#define QNN_HTP_OPTION_INIT              \
  {                                      \
    QNN_HTP_OPTION_INIT, /*htp_options*/ \
        kLogOff,         /*log_level*/   \
  }

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
