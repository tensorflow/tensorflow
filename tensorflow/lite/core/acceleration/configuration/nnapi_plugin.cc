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

// This file implements the TFLite Delegate Plugin for the NNAPI Delegate.

#include "tensorflow/lite/core/acceleration/configuration/nnapi_plugin.h"

#include "tensorflow/lite/core/acceleration/configuration/delegate_registry.h"

namespace tflite {
namespace delegates {

TFLITE_REGISTER_DELEGATE_FACTORY_FUNCTION(NnapiPlugin, NnapiPlugin::New);

}  // namespace delegates
}  // namespace tflite
