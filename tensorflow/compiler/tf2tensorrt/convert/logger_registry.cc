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
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/convert/logger_registry.h"

#include <unordered_map>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace tensorrt {

class LoggerRegistryImpl : public LoggerRegistry {
  Status Register(const string& name, nvinfer1::ILogger* logger) override {
    mutex_lock lock(mu_);
    if (!registry_.emplace(name, std::unique_ptr<nvinfer1::ILogger>(logger))
             .second) {
      return errors::AlreadyExists("Logger ", name, " already registered");
    }
    return OkStatus();
  }

  nvinfer1::ILogger* LookUp(const string& name) override {
    mutex_lock lock(mu_);
    const auto found = registry_.find(name);
    if (found == registry_.end()) {
      return nullptr;
    }
    return found->second.get();
  }

 private:
  mutable mutex mu_;
  mutable std::unordered_map<string, std::unique_ptr<nvinfer1::ILogger>>
      registry_ TF_GUARDED_BY(mu_);
};

LoggerRegistry* GetLoggerRegistry() {
  static LoggerRegistryImpl* registry = new LoggerRegistryImpl;
  return registry;
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
