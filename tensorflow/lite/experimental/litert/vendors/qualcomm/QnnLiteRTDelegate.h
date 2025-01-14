//==============================================================================
//
//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.
//
//==============================================================================
#ifndef TENSORFLOW_LITE_DELEGATES_QNN_QNN_TFLITE_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_QNN_QNN_TFLITE_DELEGATE_H_

/// Logging level of the delegate and QNN backend.
typedef enum TfLiteQnnDelegateLogLevel {  // NOLINT(modernize-use-using)
  /// Disable delegate and QNN backend logging messages.
  kLogOff = 0,
  kLogLevelError = 1,
  kLogLevelWarn = 2,
  kLogLevelInfo = 3,
  kLogLevelVerbose = 4,
  kLogLevelDebug = 5,
} TfLiteQnnDelegateLogLevel;

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

// This option can be used to specify performance mode for QNN HTP.
typedef enum TfLiteQnnDelegateHtpPerfCtrlStrategy {  // NOLINT(modernize-use-using)
  kHtpPerfCtrlManual = 0,
  kHtpPerfCtrlAuto = 1,
} TfLiteQnnDelegateHtpPerfCtrlStrategy;

// This option can be used to specify QNN HTP options.
static const char* kDispatchOptionQnnDelegateHtpBackendOptions =
    "qnn_delegate_htp_backend_options";

/// Specifies the backend options for the HTP backend.
typedef struct {  // NOLINT
  /// The default performance mode sets no configurations on the HTP.
  TfLiteQnnDelegateHtpPerformanceMode performance_mode;
  /// The default performance control strategy is Manual.
  TfLiteQnnDelegateHtpPerfCtrlStrategy perf_ctrl_strategy;
  /// Log level
  TfLiteQnnDelegateLogLevel log_level;
} TfLiteQnnDelegateHtpBackendOptions;

#define QNN_DELEGATE_HTP_OPTION_INIT               \
  {                                                \
    kHtpDefault,            /*performance_mode*/   \
        kHtpPerfCtrlManual, /*perf_ctrl_strategy*/ \
        kLogOff,            /*log_level*/          \
  }
#endif  // TENSORFLOW_LITE_DELEGATES_QNN_QNN_TFLITE_DELEGATE_H_
