/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/c/logging.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringprintf.h"

static ::tensorflow::string BuildMessage(const char* fmt, va_list args) {
  ::tensorflow::string message;
  ::tensorflow::strings::Appendv(&message, fmt, args);
  return message;
}

void TF_Log(TF_LogLevel level, const char* fmt, ...) {
  if (level < TF_INFO || level > TF_FATAL) return;
  va_list args;
  va_start(args, fmt);
  auto message = BuildMessage(fmt, args);
  va_end(args);
  switch (level) {
    case TF_INFO:
      LOG(INFO) << message;
      break;
    case TF_WARNING:
      LOG(WARNING) << message;
      break;
    case TF_ERROR:
      LOG(ERROR) << message;
      break;
    case TF_FATAL:
      LOG(FATAL) << message;
      break;
  }
}

void TF_VLog(int level, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  auto message = BuildMessage(fmt, args);
  va_end(args);
  VLOG(level) << message;
}

void TF_DVLog(int level, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  auto message = BuildMessage(fmt, args);
  va_end(args);
  DVLOG(level) << message;
}
