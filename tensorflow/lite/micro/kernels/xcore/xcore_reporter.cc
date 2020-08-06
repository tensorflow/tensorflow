// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include "tensorflow/lite/micro/kernels/xcore/xcore_reporter.h"

#include <cstdarg>

#ifndef TF_LITE_STRIP_ERROR_STRINGS
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_string.h"
#endif

namespace tflite {
namespace micro {
namespace xcore {

int XCoreReporter::Report(const char* format, va_list args) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  // Only pulling in the implementation of this function for builds where we
  // expect to make use of it to be extra cautious about not increasing the code
  // size.
  static constexpr int kMaxLogLen = 256;
  char log_buffer[kMaxLogLen];
  MicroVsnprintf(log_buffer, kMaxLogLen, format, args);
  DebugLog(log_buffer);
  DebugLog("\r\n");
#endif
  return 0;
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite