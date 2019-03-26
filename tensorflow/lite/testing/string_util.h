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
#ifndef TENSORFLOW_LITE_TESTING_STRING_UTIL_H_
#define TENSORFLOW_LITE_TESTING_STRING_UTIL_H_

#include <Python.h>
#include <string>

namespace tflite {
namespace testing {
namespace python {

// Take a python string array, convert it to TF Lite dynamic buffer format and
// serialize it as a HexString.
PyObject* SerializeAsHexString(PyObject* value);

}  // namespace python
}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_STRING_UTIL_H_
