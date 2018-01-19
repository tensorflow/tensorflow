// -*- c++ -*-
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_TENSORRT_LOG_TRT_LOGGER_H_
#define TENSORFLOW_CONTRIB_TENSORRT_LOG_TRT_LOGGER_H_

// Use TF logging f
#include <NvInfer.h>
#include <string>

//------------------------------------------------------------------------------
namespace tensorflow {

//------------------------------------------------------------------------------
namespace tensorrt {

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity, const char* msg) override;

 private:
  std::string name_;
};

}  // namespace tensorrt

}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_TENSORRT_LOG_TRT_LOGGER_H_
