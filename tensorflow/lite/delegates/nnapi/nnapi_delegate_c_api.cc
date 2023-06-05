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

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

TfLiteDelegate* TfLiteNnapiDelegateCreate(
    const TfLiteNnapiDelegateOptions* options) {
  tflite::StatefulNnApiDelegate::StatefulNnApiDelegate::Options
      internal_options;
  internal_options.execution_preference =
      static_cast<tflite::StatefulNnApiDelegate::StatefulNnApiDelegate::
                      Options::ExecutionPreference>(
          options->execution_preference);
  internal_options.accelerator_name = options->accelerator_name;
  internal_options.cache_dir = options->cache_dir;
  internal_options.model_token = options->model_token;
  internal_options.disallow_nnapi_cpu = options->disallow_nnapi_cpu;
  internal_options.max_number_delegated_partitions =
      options->max_number_delegated_partitions;
  internal_options.allow_fp16 = options->allow_fp16;

  tflite::StatefulNnApiDelegate* delegate = nullptr;
  if (options->nnapi_support_library_handle) {
    delegate = new tflite::StatefulNnApiDelegate(
        static_cast<NnApiSLDriverImplFL5*>(
            options->nnapi_support_library_handle),
        internal_options);
  } else {
    delegate = new tflite::StatefulNnApiDelegate(internal_options);
  }
  return delegate;
}

TfLiteNnapiDelegateOptions TfLiteNnapiDelegateOptionsDefault() {
  TfLiteNnapiDelegateOptions result = {};
  tflite::StatefulNnApiDelegate::Options options;
  result.execution_preference =
      static_cast<TfLiteNnapiDelegateOptions::ExecutionPreference>(
          options.execution_preference);
  result.accelerator_name = options.accelerator_name;
  result.cache_dir = options.cache_dir;
  result.model_token = options.model_token;
  result.disallow_nnapi_cpu = options.disallow_nnapi_cpu;
  result.max_number_delegated_partitions =
      options.max_number_delegated_partitions;
  result.allow_fp16 = options.allow_fp16;
  result.nnapi_support_library_handle = nullptr;
  return result;
}

void TfLiteNnapiDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate == nullptr) return;
  delete static_cast<tflite::StatefulNnApiDelegate*>(delegate);
}
