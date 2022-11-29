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
#ifndef TENSORFLOW_LITE_C_C_API_OPAQUE_INTERNAL_H_
#define TENSORFLOW_LITE_C_C_API_OPAQUE_INTERNAL_H_

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace internal {

class CommonOpaqueConversionUtil {
 public:
  // Create a 'TfLiteRegistrationExternal' object that corresponds to the
  // provided 'registration' argument, set it as the 'registration's
  // 'registration_external' field and return the address of the external
  // registration.  We loosely define that a 'TfLiteRegistrationExternal' object
  // "corresponds" to a 'TfLiteRegistration' object when calling any function
  // pointer (like 'prepare') on the 'TfLiteRegistrationExternal' object calls
  // into the corresponding function pointer of the 'TfLiteRegistration' object.
  //
  // The specified 'context' is used to store the 'TfLiteRegistrationExternal*'
  // pointers. The 'TfLiteRegistrationExternal*' pointer will be deallocated
  // when the 'context' gets destroyed.  I.e., the caller of this function
  // should not deallocate the object pointed to by the return value of
  // 'ObtainRegistrationExternal'.
  static TfLiteRegistrationExternal* ObtainRegistrationExternal(
      TfLiteContext* context, TfLiteRegistration* registration);
};
}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_C_C_API_OPAQUE_INTERNAL_H_
