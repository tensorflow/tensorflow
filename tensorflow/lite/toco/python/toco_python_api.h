/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOCO_PYTHON_TOCO_PYTHON_API_H_
#define TENSORFLOW_LITE_TOCO_PYTHON_TOCO_PYTHON_API_H_

#include <Python.h>

namespace toco {

// Convert a model represented in `input_contents`. `model_flags_proto`
// describes model parameters. `toco_flags_proto` describes conversion
// parameters (see relevant .protos for more information). Returns a string
// representing the contents of the converted model. When extended_return
// flag is set to true returns a dictionary that contains string representation
// of the converted model and some statistics like arithmetic ops count.
PyObject* TocoConvert(PyObject* model_flags_proto_txt_raw,
                      PyObject* toco_flags_proto_txt_raw,
                      PyObject* input_contents_txt_raw,
                      bool extended_return = false);

// All the exported functions should be listed in
// tensorflow/tools/def_file_filter/symbols_pybind.txt for the Windows build.
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_PYTHON_TOCO_PYTHON_API_H_
