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

/* Wrap trt_conversion */
%{
#define SWIG_FILE_WITH_INIT
%}

%{
struct version_struct{
  int vmajor;
  int vminor;
  int vpatch;
};

PyObject* version_helper(version_struct* in) {
  PyObject *tuple(nullptr);
  tuple = Py_BuildValue("(iii)", in->vmajor, in->vminor, in->vpatch);
  if (!tuple) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError,
                      "Tuple creation from version structure failed!");
    }
    return NULL;
  }
  return tuple;
}

%}

%typemap(out) version_struct {
  PyObject *tuple = version_helper(&$1);
  if (!tuple) SWIG_fail;
  $result = tuple;
}

%{
#include "tensorflow/compiler/tf2tensorrt/utils/py_utils.h"
%}

%ignore "";
%rename("%s") get_linked_tensorrt_version;
%rename("%s") get_loaded_tensorrt_version;
%rename("%s") is_tensorrt_enabled;

%{

version_struct get_linked_tensorrt_version() {
  // Return the version at the link time.
  version_struct s;
  tensorflow::tensorrt::GetLinkedTensorRTVersion(
      &s.vmajor, &s.vminor, &s.vpatch);
  return s;
}

version_struct get_loaded_tensorrt_version() {
  // Return the version from the loaded library.
  version_struct s;
  tensorflow::tensorrt::GetLoadedTensorRTVersion(
      &s.vmajor, &s.vminor, &s.vpatch);
  return s;
}

bool is_tensorrt_enabled() {
  return tensorflow::tensorrt::IsGoogleTensorRTEnabled();
}

%}

version_struct get_linked_tensorrt_version();
version_struct get_loaded_tensorrt_version();
bool is_tensorrt_enabled();

%rename("%s") "";
