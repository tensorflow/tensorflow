/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <utility>

#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/stable_delegate_interface.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace {

TfLiteOpaqueDelegate* SampleStableDelegateCreateFunc(
    const void* tflite_settings) {
  auto delegate = std::make_unique<tflite::example::SampleStableDelegate>();
  return tflite::TfLiteOpaqueDelegateFactory::CreateSimpleDelegate(
      std::move(delegate));
}

void SampleStableDelegateDestroyFunc(
    TfLiteOpaqueDelegate* sample_stable_delegate) {
  tflite::TfLiteOpaqueDelegateFactory::DeleteSimpleDelegate(
      sample_stable_delegate);
}

int SampleStableDelegateErrnoFunc(
    TfLiteOpaqueDelegate* sample_stable_delegate) {
  // no-op
  return 0;
}

const TfLiteOpaqueDelegatePlugin sample_stable_delegate_plugin = {
    SampleStableDelegateCreateFunc, SampleStableDelegateDestroyFunc,
    SampleStableDelegateErrnoFunc};

const TfLiteStableDelegate sample_stable_delegate = {
    TFL_STABLE_DELEGATE_ABI_VERSION, tflite::example::kSampleStableDelegateName,
    tflite::example::kSampleStableDelegateVersion,
    &sample_stable_delegate_plugin};

}  // namespace

/**
 * A super simple test delegate for testing.
 */
extern "C" const TfLiteStableDelegate TFL_TheStableDelegate =
    sample_stable_delegate;
