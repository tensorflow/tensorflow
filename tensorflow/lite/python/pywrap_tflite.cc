/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
/*
This is a pywrap is for the pybind_extension to allow any c++ to be used
in python py_library or py_test. Please add any c++ runtime dependency
decoupled from pywrap_tensorflow in deps field under
tensorflow/lite/python:pywrap_tflite for the #tfsplit effort
*/

#include "pybind11/pybind11.h"  // from @pybind11

// This logic allows Python to import _pywrap_tflite.so by
// creating a PyInit function and exposing it. It is required in opensource
// only.
PYBIND11_MODULE(_pywrap_tflite, m){};
