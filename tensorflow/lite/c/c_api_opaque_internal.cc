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
#include "tensorflow/lite/c/c_api_opaque_internal.h"

#include <memory>
#include <unordered_map>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"

namespace tflite {
namespace internal {

static constexpr char kDataNullLog[] =
    "The supplied 'data' argument must not be null.";

TfLiteRegistrationExternal*
CommonOpaqueConversionUtil::ObtainRegistrationExternal(
    TfLiteContext* context, TfLiteRegistration* registration, int node_index) {
  // We need to allocate a new TfLiteRegistrationExternal object and then
  // populate its state correctly, based on the contents in 'registration'.

  auto* registration_external = TfLiteRegistrationExternalCreate(
      static_cast<TfLiteBuiltinOperator>(registration->builtin_code),
      registration->custom_name, registration->version);

  registration_external->node_index = node_index;

  registration->registration_external = registration_external;

  auto* subgraph = static_cast<tflite::Subgraph*>(context->impl_);
  subgraph->registration_externals_.insert(
      std::unique_ptr<TfLiteRegistrationExternal>(registration_external));
  return registration_external;
}
}  // namespace internal
}  // namespace tflite
