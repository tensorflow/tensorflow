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

#ifndef TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_BUILDER_H_
#define TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_BUILDER_H_

#include <memory>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/status.h"

namespace tensorflow {
namespace experimental {
namespace cc {

// RuntimeBuilder is a builder used to construct a tensorflow::cc::Runtime.
// Use this to set configuration options, like threadpool size, etc.
class RuntimeBuilder {
 public:
  RuntimeBuilder() : options_(TFE_NewContextOptions()) {}

  // If `use_tfrt` is true, we will use the new Tensorflow Runtime
  // (https://blog.tensorflow.org/2020/04/tfrt-new-tensorflow-runtime.html) as
  // our runtime implementation.
  RuntimeBuilder& SetUseTFRT(bool use_tfrt);

  // Build a Tensorflow Runtime.
  //
  // Params:
  //  status - Set to OK on success and an appropriate error on failure.
  // Returns:
  //  If status is not OK, returns nullptr. Otherwise, returns a
  //  unique_ptr<tensorflow::cc::Runtime>.
  std::unique_ptr<Runtime> Build(Status* status);

  // RuntimeBuilder is movable, but not copyable.
  RuntimeBuilder(RuntimeBuilder&&) = default;
  RuntimeBuilder& operator=(RuntimeBuilder&&) = default;

 private:
  // RuntimeBuilder is not copyable
  RuntimeBuilder(const RuntimeBuilder&) = delete;
  RuntimeBuilder& operator=(const RuntimeBuilder&) = delete;

  struct TFEContextOptionsDeleter {
    void operator()(TFE_ContextOptions* p) const {
      TFE_DeleteContextOptions(p);
    }
  };
  std::unique_ptr<TFE_ContextOptions, TFEContextOptionsDeleter> options_;
};

inline RuntimeBuilder& RuntimeBuilder::SetUseTFRT(bool use_tfrt) {
  TFE_ContextOptionsSetTfrt(options_.get(), use_tfrt);
  return *this;
}

inline std::unique_ptr<Runtime> RuntimeBuilder::Build(Status* status) {
  TFE_Context* result = TFE_NewContext(options_.get(), status->GetTFStatus());
  if (!status->ok()) {
    return nullptr;
  }
  // We can't use std::make_unique here because of its interaction with a
  // private constructor: https://abseil.io/tips/134
  return std::unique_ptr<Runtime>(new Runtime(result));
}

}  // namespace cc
}  // namespace experimental
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_EXPERIMENTAL_BASE_PUBLIC_RUNTIME_BUILDER_H_
