/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/tensorrt/log/trt_logger.h"

#include "tensorflow/core/platform/logging.h"

// Use TF logging for TensorRT informations

#define _TF_LOG_DEBUG ::tensorflow::internal::LogMessage(__FILE__, __LINE__, -1)

namespace tensorflow {
namespace tensorrt {

void Logger::log(Severity severity, const char* msg) {
  // Suppress info-level messages
  switch (severity) {
    case Severity::kINFO: {  // mark TRT info messages as debug!
      VLOG(-1) << msg;
    case Severity::kINFO: {  // Mark TRT info messages as debug!
      LOG(DEBUG) << msg;
      break;
    }
    case Severity::kWARNING: {
      LOG(WARNING) << msg;
      break;
    }
    case Severity::kERROR: {
      LOG(ERROR) << msg;
      break;
    }
    case Severity::kINTERNAL_ERROR: {
      LOG(FATAL) << msg;
      break;
    }
    // This is useless for now. But would catch it in future if enum changes. It
    // is always good to have default case!
    default: {
      LOG(FATAL) << name_ << "Got unknown severity level from TRT " << msg;
      break;
    }
  }
}

}  // namespace tensorrt
}  // namespace tensorflow
