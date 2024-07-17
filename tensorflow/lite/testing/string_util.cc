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

#include "tensorflow/lite/testing/string_util.h"

#include <memory>

#include "absl/strings/escaping.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace testing {
namespace python {

PyObject* SerializeAsHexString(PyObject* value) {
  DynamicBuffer dynamic_buffer;
  if (!python_utils::FillStringBufferWithPyArray(value, &dynamic_buffer)) {
    return nullptr;
  }

  char* char_buffer = nullptr;
  size_t size = dynamic_buffer.WriteToBuffer(&char_buffer);
  string s = absl::BytesToHexString({char_buffer, size});
  free(char_buffer);

  return python_utils::ConvertToPyString(s.data(), s.size());
}

}  // namespace python
}  // namespace testing
}  // namespace tflite
