// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_QNN_LITERT_DELEGATE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_QNN_LITERT_DELEGATE_H_
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/common.h"

/// Defines performance modes available for HTP backend.
typedef enum TfLiteQnnDelegateHtpPerformanceMode {  // NOLINT(modernize-use-using)
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
} TfLiteQnnDelegateHtpPerformanceMode;

/// Specifies the backend options for the HTP backend.
typedef struct {  // NOLINT
  /// The default performance mode sets no configurations on the HTP.
  TfLiteQnnDelegateHtpPerformanceMode performance_mode;
} TfLiteQnnDelegateHtpBackendOptions;

// This option can be used to specify QNN options.
static const char* kDispatchOptionQnnDelegateOptions = "qnn_delegate_options";

/// Specifies the backend options for the HTP backend.
typedef struct {  // NOLINT
  /// Optional backend specific options for the HTP backend.
  TfLiteQnnDelegateHtpBackendOptions htp_options;
  /// Log level
  LiteRtQnnLogLevel log_level;
} TfLiteQnnDelegateOptions;

#define QNN_DELEGATE_HTP_OPTION_INIT {kHtpDefault /*performance_mode*/}
#define QNN_DELEGATE_OPTION_INIT                    \
  {                                                 \
      QNN_DELEGATE_HTP_OPTION_INIT, /*htp_options*/ \
      kLogOff,                      /*log_level*/   \
  }
#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_QNN_LITERT_DELEGATE_H_
