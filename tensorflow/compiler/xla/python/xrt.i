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

// Wrappers for XRT ops.

%module(threads="1") xrt

// Keep the GIL except where explicitly specified.
%nothread;

%include "tensorflow/python/platform/base.i"
%include "tensorflow/compiler/xla/python/xla_data.i"

%{
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/xrt.h"

using namespace xla;
using namespace xla::swig;

%}

// Computation and buffer/allocation types

%typemap(out) StatusOr<xla::swig::XrtExecutable*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::XrtExecutable*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::XrtAllocation*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::XrtAllocation*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::XrtAllocationTuple*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::XrtAllocationTuple*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}


%typemap(in) absl::Span<xla::swig::XrtAllocation* const>
    (std::vector<XrtAllocation*> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    XrtAllocation* xrta;
    if ((SWIG_ConvertPtr(o, (void**) &xrta, $descriptor(xla::swig::XrtAllocation*),
                         SWIG_POINTER_EXCEPTION)) == -1) {
      SWIG_fail;
    }
    temps.push_back(xrta);
    Py_DECREF(o);
  }
  $1 = temps;
}


%ignoreall
%unignore xla;
%unignore xla::swig;
%unignore xla::swig::XrtAllocation;
%unignore xla::swig::XrtAllocation::FromLiteral;
%unignore xla::swig::XrtAllocation::ToLiteral;
%unignore xla::swig::XrtAllocation::shape;
%unignore xla::swig::XrtAllocationTuple;
%unignore xla::swig::XrtAllocationTuple::Release;
%unignore xla::swig::XrtAllocationTuple::size;
%unignore xla::swig::XrtExecutable;
%unignore xla::swig::XrtExecutable::CompileForXrt;
%unignore xla::swig::XrtExecutable::DeviceOrdinals;
%unignore xla::swig::XrtExecutable::Execute;
%unignore xla::swig::DestructureXrtAllocationTuple;
%unignore xla::swig::DeleteXrtAllocation;
%unignore xla::swig::DeleteXrtExecutable;

%thread;
%include "tensorflow/compiler/xla/python/xrt.h"
%nothread;

%unignoreall
