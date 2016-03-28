/* Copyright 2016 Google Inc. All Rights Reserved.

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

%nothread tensorflow::ServerInterface::Join;

%include "tensorflow/python/platform/base.i"

//%newobject tensorflow::NewServer;

%typemap(in) const ServerDef& (tensorflow::ServerDef temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The ServerDef could not be parsed as a valid protocol buffer");
    SWIG_fail;
  }
  $1 = &temp;
}

%typemap(in, numinputs=0)
    std::unique_ptr<tensorflow::ServerInterface>* out_server (
        std::unique_ptr<tensorflow::ServerInterface> temp) {
  $1 = &temp;
}

%typemap(out) tensorflow::Status tensorflow::NewServer {
  if (!$1.ok()) {
    RaiseStatusNotOK($1, $descriptor(tensorflow::Status*));
    SWIG_fail;
  }
} 

%typemap(argout) std::unique_ptr<tensorflow::ServerInterface>* out_server {
  // TODO(mrry): Convert this to SWIG_POINTER_OWN when the issues with freeing
  // a server are fixed.
  $result = SWIG_NewPointerObj($1->release(),
                               $descriptor(tensorflow::ServerInterface*),
                               0);
}

%feature("except") tensorflow::ServerInterface::Join {
  // Let other threads run while we wait for the server to shut down.
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%{
#include "tensorflow/core/distributed_runtime/server_lib.h"

using tensorflow::ServerDef;
%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::ServerDef;
%unignore tensorflow::ServerInterface;
%unignore tensorflow::ServerInterface::~ServerInterface;
%unignore tensorflow::ServerInterface::Start;
%unignore tensorflow::ServerInterface::Stop;
%unignore tensorflow::ServerInterface::Join;
%unignore tensorflow::ServerInterface::target;

%unignore tensorflow::NewServer;

%include "tensorflow/core/distributed_runtime/server_lib.h"

%unignoreall
