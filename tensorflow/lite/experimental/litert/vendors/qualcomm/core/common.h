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

typedef struct {  // NOLINT(modernize-use-using)
  /// Apply HTP-friendly op builder.
  bool useHtpPreferencs;
  /// This option will treat quantized int16 tensor as quantized uint16 tensor
  /// for better backend compatibility.
  bool useQInt16AsQUint16;
} LiteRtQnnOptions;

// clang-format off
#define LITERT_QNN_OPTIONS_INIT      \
  {                                  \
    false,    /*useHtpPreferencs*/   \
    true,     /*useQInt16AsQUint16*/ \
  }
// clang-format on
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_COMMON_H_
