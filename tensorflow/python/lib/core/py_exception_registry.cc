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

#include "tensorflow/python/lib/core/py_exception_registry.h"

#include <Python.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

PyExceptionRegistry* PyExceptionRegistry::singleton_ = nullptr;

void PyExceptionRegistry::Init(PyObject* code_to_exc_type_map) {
  CHECK(singleton_ == nullptr) << "PyExceptionRegistry::Init() already called";
  singleton_ = new PyExceptionRegistry;

  CHECK(PyDict_Check(code_to_exc_type_map));
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(code_to_exc_type_map, &pos, &key, &value)) {
    singleton_->exc_types_.emplace(static_cast<TF_Code>(PyLong_AsLong(key)),
                                   value);
    // The exception classes should also have the lifetime of the process, but
    // incref just in case.
    Py_INCREF(value);
  }

  static const TF_Code kAllCodes[] = {TF_CANCELLED,
                                      TF_UNKNOWN,
                                      TF_INVALID_ARGUMENT,
                                      TF_DEADLINE_EXCEEDED,
                                      TF_NOT_FOUND,
                                      TF_ALREADY_EXISTS,
                                      TF_PERMISSION_DENIED,
                                      TF_UNAUTHENTICATED,
                                      TF_RESOURCE_EXHAUSTED,
                                      TF_FAILED_PRECONDITION,
                                      TF_ABORTED,
                                      TF_OUT_OF_RANGE,
                                      TF_UNIMPLEMENTED,
                                      TF_INTERNAL,
                                      TF_UNAVAILABLE,
                                      TF_DATA_LOSS};
  for (TF_Code code : kAllCodes) {
    CHECK(singleton_->exc_types_.find(code) != singleton_->exc_types_.end())
        << error::Code_Name(static_cast<error::Code>(code))
        << " is not registered";
  }
}

PyObject* PyExceptionRegistry::Lookup(TF_Code code) {
  CHECK(singleton_ != nullptr) << "Must call PyExceptionRegistry::Init() "
                                  "before PyExceptionRegistry::Lookup()";
  CHECK_NE(code, TF_OK);
  auto it = singleton_->exc_types_.find(code);
  CHECK(it != singleton_->exc_types_.end())
      << "Unknown error code passed to PyExceptionRegistry::Lookup: " << code;
  return it->second;
}

}  // namespace tensorflow
