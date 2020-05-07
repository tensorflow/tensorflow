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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_LOGGER_REGISTRY_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_LOGGER_REGISTRY_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

class LoggerRegistry {
 public:
  virtual Status Register(const string& name, nvinfer1::ILogger* logger) = 0;
  virtual nvinfer1::ILogger* LookUp(const string& name) = 0;
  virtual ~LoggerRegistry() {}
};

LoggerRegistry* GetLoggerRegistry();

class RegisterLogger {
 public:
  RegisterLogger(const string& name, nvinfer1::ILogger* logger) {
    TF_CHECK_OK(GetLoggerRegistry()->Register(name, logger));
  }
};

#define REGISTER_TENSORRT_LOGGER(name, logger) \
  REGISTER_TENSORRT_LOGGER_UNIQ_HELPER(__COUNTER__, name, logger)
#define REGISTER_TENSORRT_LOGGER_UNIQ_HELPER(ctr, name, logger) \
  REGISTER_TENSORRT_LOGGER_UNIQ(ctr, name, logger)
#define REGISTER_TENSORRT_LOGGER_UNIQ(ctr, name, logger)                 \
  static ::tensorflow::tensorrt::RegisterLogger register_trt_logger##ctr \
      TF_ATTRIBUTE_UNUSED =                                              \
          ::tensorflow::tensorrt::RegisterLogger(name, logger)

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_LOGGER_REGISTRY_H_
