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
#include "tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.h"

#include <utility>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace dummy_test {

// Dummy delegate kernel.
class DummyDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit DummyDelegateKernel(const DummyDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  const DummyDelegateOptions options_;
};

// DummyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class DummyDelegate : public SimpleDelegateInterface {
 public:
  explicit DummyDelegate(const DummyDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    return options_.allowed_builtin_code == registration->builtin_code;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "DummyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<DummyDelegateKernel>(options_);
  }

 private:
  const DummyDelegateOptions options_;
};

}  // namespace dummy_test
}  // namespace tflite

DummyDelegateOptions TfLiteDummyDelegateOptionsDefault() {
  DummyDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this dummy test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteDummyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteDummyDelegateCreate(const DummyDelegateOptions* options) {
  std::unique_ptr<tflite::dummy_test::DummyDelegate> dummy(
      new tflite::dummy_test::DummyDelegate(
          options ? *options : TfLiteDummyDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(dummy));
}

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
