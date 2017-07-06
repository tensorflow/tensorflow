/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/profiler/internal/print_model_analysis.h"

using tensorflow::int64;
%}

%typemap(typecheck) const string & = char *;
%typemap(in) const string& (string temp) {
  if (!_PyObjAs<string>($input, &temp)) return NULL;
  $1 = &temp;
}
%typemap(out) const string& {
%#if PY_MAJOR_VERSION >= 3
  $result = PyUnicode_FromStringAndSize($1->data(), $1->size());
%#else
  $result = PyString_FromStringAndSize($1->data(), $1->size());
%#endif
}
%apply const string & {string &};
%apply const string & {string *};

%ignoreall

%unignore tensorflow;
%unignore tensorflow::tfprof;
%unignore tensorflow::tfprof::PrintModelAnalysis;
%unignore tensorflow::tfprof::NewProfiler;
%unignore tensorflow::tfprof::DeleteProfiler;
%unignore tensorflow::tfprof::AddStep;
%unignore tensorflow::tfprof::Profile;

%include "tensorflow/core/profiler/internal/print_model_analysis.h"

%unignoreall