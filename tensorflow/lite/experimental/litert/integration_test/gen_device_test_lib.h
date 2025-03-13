// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_INTEGRATION_TEST_GEN_DEVICE_TEST_LIB_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_INTEGRATION_TEST_GEN_DEVICE_TEST_LIB_H_

namespace litert::test {

// Absract wrapper for the invocation of the compiled model API within a
// standard test environment.
class CmInvoker {
 public:
  using Ptr = std::unique_ptr<CmInvoker>;

  CmInvoker(Environment&& env, Model&& model)
      : env_(std::move(env)), model_(std::move(model)) {}

  // Setup the compiled model api and initialize the input and output buffers.
  // Assumes default signature.
  void Setup() {
    LITERT_ASSERT_OK_AND_ASSIGN(
        compiled_model_, CompiledModel::Create(env_, model_, Accelerator()));
    const auto sig = model_.DefaultSignatureKey();
    LITERT_ASSERT_OK_AND_ASSIGN(input_buffers_,
                                compiled_model_.CreateInputBuffers(sig));
    LITERT_ASSERT_OK_AND_ASSIGN(output_buffers_,
                                compiled_model_.CreateOutputBuffers(sig));
  }

  // Invoke the compiled model api. Must be called after Setup().
  void Run() {
    ASSERT_TRUE(compiled_model_.Run(model_.DefaultSignatureKey(),
                                    input_buffers_, output_buffers_));
  }

  // Is this test in a state where it should be skipped? Implementations should
  // call GTEST_SKIP().
  virtual void MaybeSkip() const = 0;

  // Which accelerator option to use.
  virtual LiteRtHwAccelerators Accelerator() const = 0;

  std::vector<TensorBuffer>& GetInputBuffers() { return input_buffers_; }
  std::vector<TensorBuffer>& GetOutputBuffers() { return output_buffers_; }

  virtual ~CmInvoker() = default;

 protected:
  Environment env_;
  Model model_;

  CompiledModel compiled_model_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
};

// Invocation of the compiled model API for the NPU accelerator. This handles
// both JIT and pre-compiled models.
class CmNpuInvoker : public CmInvoker {
 public:
  CmNpuInvoker(Environment&& env, Model&& model)
      : CmInvoker(std::move(env), std::move(model)) {}

  // Will invocation require compilation.
  bool IsJit() const {
    auto& m = *model_.Get();
    return !IsCompiled(m);
  }

  LiteRtHwAccelerators Accelerator() const override {
    return IsJit() ? kLiteRtHwAcceleratorNpu : kLiteRtHwAcceleratorNone;
  }

  void MaybeSkip() const override {
#if !defined(__ANDROID__)
    GTEST_SKIP() << "NPU test must run on android device.";
#endif
  }
};

// Invocation of the compiled model API on CPU. This can run on linux in
// addition to android.
class CmCpuInvoker : public CmInvoker {
 public:
  CmCpuInvoker(Environment&& env, Model&& model)
      : CmInvoker(std::move(env), std::move(model)) {}

  LiteRtHwAccelerators Accelerator() const override {
    return kLiteRtHwAcceleratorCpu;
  }

  void MaybeSkip() const override {
    if (IsCompiled(*model_.Get())) {
      GTEST_SKIP() << "Cannot run CPU test on a compiled model.";
    }
  }
};

}  // namespace litert::test

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_INTEGRATION_TEST_GEN_DEVICE_TEST_LIB_H_
