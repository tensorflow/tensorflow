/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_
#define TENSORFLOW_CONTRIB_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_

// These functions transform codes and data structures that are defined in the
// flatbuffer serialization format into in-memory values that are used by the
// runtime API and interpreter.

#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/core/api/error_reporter.h"
#include "tensorflow/contrib/lite/core/api/op_resolver.h"
#include "tensorflow/contrib/lite/schema/schema_generated.h"

namespace tflite {

// Interface class for builtin data allocations.
class BuiltinDataAllocator {
 public:
  virtual void* Allocate(size_t size) = 0;
  virtual void Deallocate(void* data) = 0;

  // Allocate a structure, but make sure it is a POD structure that doesn't
  // require constructors to run. The reason we do this, is that Interpreter's C
  // extension part will take ownership so destructors  will not be run during
  // deallocation.
  template <typename T>
  T* AllocatePOD() {
    static_assert(std::is_pod<T>::value, "Builtin data structure must be POD.");
    return static_cast<T*>(this->Allocate(sizeof(T)));
  }

  virtual ~BuiltinDataAllocator() {}
};

// Parse the appropriate data out of the op.
//
// This handles builtin data explicitly as there are flatbuffer schemas.
// If it returns kTfLiteOk, it passes the data out with `builtin_data`. The
// calling function has to pass in an allocator object, and this allocator
// will be called to reserve space for the output data. If the calling
// function's allocator reserves memory on the heap, then it's the calling
// function's responsibility to free it.
// If it returns kTfLiteError, `builtin_data` will be `nullptr`.
TfLiteStatus ParseOpData(const Operator* op, BuiltinOperator op_type,
                         ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data);

// Converts the tensor data type used in the flat buffer to the representation
// used by the runtime.
TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type,
                               ErrorReporter* error_reporter);

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_CORE_API_FLATBUFFER_CONVERSIONS_H_
