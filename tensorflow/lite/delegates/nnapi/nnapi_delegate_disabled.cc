/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

namespace tflite {

// Return a non-functional NN API Delegate struct.
TfLiteDelegate* NnApiDelegate() {
  static StatefulNnApiDelegate* delegate = new StatefulNnApiDelegate();
  return delegate;
}

StatefulNnApiDelegate::StatefulNnApiDelegate(const NnApi* /* nnapi */)
    : StatefulNnApiDelegate() {}

StatefulNnApiDelegate::StatefulNnApiDelegate(Options /* options */)
    : StatefulNnApiDelegate() {}

StatefulNnApiDelegate::StatefulNnApiDelegate(const NnApi* /* nnapi */,
                                             Options /* options */)
    : StatefulNnApiDelegate() {}

StatefulNnApiDelegate::StatefulNnApiDelegate(
    const NnApiSLDriverImplFL5* /* nnapi_support_library_driver */,
    Options /* options */)
    : StatefulNnApiDelegate() {}

StatefulNnApiDelegate::StatefulNnApiDelegate()
    : TfLiteDelegate(TfLiteDelegateCreate()),
      delegate_data_(/*nnapi=*/nullptr) {
  Prepare = DoPrepare;
}

TfLiteStatus StatefulNnApiDelegate::DoPrepare(TfLiteContext* /* context */,
                                              TfLiteDelegate* /* delegate */) {
  return kTfLiteOk;
}

TfLiteBufferHandle StatefulNnApiDelegate::RegisterNnapiMemory(
    ANeuralNetworksMemory* memory, CopyToHostTensorFnPtr callback,
    void* callback_context) {
  return kTfLiteNullBufferHandle;
}

int StatefulNnApiDelegate::GetNnApiErrno() const { return 0; }

using ::tflite::delegate::nnapi::NNAPIDelegateKernel;

StatefulNnApiDelegate::Data::Data(const NnApi* nnapi) : nnapi(nnapi) {}

StatefulNnApiDelegate::Data::~Data() {}

void StatefulNnApiDelegate::Data::CacheDelegateKernel(
    const TfLiteDelegateParams* delegate_params,
    NNAPIDelegateKernel* delegate_state) {}

NNAPIDelegateKernel* StatefulNnApiDelegate::Data::MaybeGetCachedDelegateKernel(
    const TfLiteDelegateParams* delegate_params) {
  return nullptr;
}

}  // namespace tflite
