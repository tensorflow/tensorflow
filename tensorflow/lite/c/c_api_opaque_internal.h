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

#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/c/common.h"

// Internal structures and subroutines used by the C API. These are likely to
// change and should not be depended on directly by any C API clients.
//
// NOTE: This header does not follow C conventions and does not define a C API.
// It is effectively an (internal) implementation detail of the C API.

namespace tflite {
namespace internal {

class CommonOpaqueConversionUtil {
 public:
  CommonOpaqueConversionUtil() = delete;

  // Obtain (or create) a 'TfLiteOperator' object that corresponds
  // to the provided 'registration' argument, and return the address of the
  // external registration.  We loosely define that a
  // 'TfLiteOperator' object "corresponds" to a 'TfLiteRegistration'
  // object when calling any function pointer (like 'prepare') on the
  // 'TfLiteOperator' object calls into the corresponding function
  // pointer of the 'TfLiteRegistration' object.
  //
  // The specified 'context' or 'op_resolver' object is used to store the
  // 'TfLiteOperator*' pointers. The 'TfLiteOperator*'
  // pointer will be deallocated when that object gets destroyed.  I.e., the
  // caller of this function should not deallocate the object pointed to by the
  // return value of 'ObtainOperator'.
  //
  // We also need to provide the 'node_index' that the 'registration'
  // corresponds to, so that the 'TfLiteOperator' can store that
  // index within its fields.  If the registration does not yet correspond
  // to a specific node index, then 'node_index' should be -1.
  static TfLiteOperator* ObtainOperator(TfLiteContext* context,
                                        const TfLiteRegistration* registration,
                                        int node_index);

 private:
  static TfLiteOperator* CachedObtainOperator(
      ::tflite::internal::OperatorsCache* registration_externals_cache,
      const TfLiteRegistration* registration, int node_index);
};

}  // namespace internal
}  // namespace tflite
#endif  // TENSORFLOW_LITE_C_C_API_OPAQUE_INTERNAL_H_
