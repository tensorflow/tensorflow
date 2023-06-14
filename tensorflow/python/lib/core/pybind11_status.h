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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_
#define TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_

#include <Python.h>

#include "pybind11/cast.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"

namespace tsl {

namespace internal {

inline PyObject* CodeToPyExc(const int code) {
  switch (code) {
    case error::Code::INVALID_ARGUMENT:
      return PyExc_ValueError;
    case error::Code::OUT_OF_RANGE:
      return PyExc_IndexError;
    case error::Code::UNIMPLEMENTED:
      return PyExc_NotImplementedError;
    default:
      return PyExc_RuntimeError;
  }
}

inline PyObject* StatusToPyExc(const Status& status) {
  return CodeToPyExc(status.raw_code());
}

inline PyObject* TFStatusToPyExc(const TF_Status* status) {
  return CodeToPyExc(TF_GetCode(status));
}

inline pybind11::dict StatusPayloadToDict(const Status& status) {
  pybind11::dict dict;
  const auto& payloads = errors::GetPayloads(status);
  for (auto& pair : payloads) {
    dict[PyBytes_FromString(pair.first.c_str())] =
        PyBytes_FromString(pair.second.c_str());
  }
  return dict;
}

inline pybind11::dict TFStatusPayloadToDict(TF_Status* status) {
  return StatusPayloadToDict(status->status);
}

}  // namespace internal

inline void MaybeRaiseFromStatus(const Status& status) {
  if (!status.ok()) {
    PyErr_SetString(internal::StatusToPyExc(status),
                    tsl::NullTerminatedMessage(status));
    throw pybind11::error_already_set();
  }
}

inline void SetRegisteredErrFromStatus(const tensorflow::Status& status) {
  PyErr_SetObject(
      tensorflow::PyExceptionRegistry::Lookup(status.raw_code()),
      pybind11::make_tuple(pybind11::none(), pybind11::none(), status.message(),
                           internal::StatusPayloadToDict(status))
          .ptr());
}

inline void SetRegisteredErrFromTFStatus(TF_Status* status) {
  PyErr_SetObject(tensorflow::PyExceptionRegistry::Lookup(TF_GetCode(status)),
                  pybind11::make_tuple(pybind11::none(), pybind11::none(),
                                       TF_Message(status),
                                       internal::TFStatusPayloadToDict(status))
                      .ptr());
}

inline void MaybeRaiseRegisteredFromStatus(const tensorflow::Status& status) {
  if (!status.ok()) {
    SetRegisteredErrFromStatus(status);
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseRegisteredFromStatusWithGIL(
    const tensorflow::Status& status) {
  if (!status.ok()) {
    // Acquire GIL for throwing exception.
    pybind11::gil_scoped_acquire acquire;
    SetRegisteredErrFromStatus(status);
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseFromTFStatus(TF_Status* status) {
  TF_Code code = TF_GetCode(status);
  if (code != TF_OK) {
    PyErr_SetString(internal::TFStatusToPyExc(status), TF_Message(status));
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseRegisteredFromTFStatus(TF_Status* status) {
  TF_Code code = TF_GetCode(status);
  if (code != TF_OK) {
    SetRegisteredErrFromTFStatus(status);
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseRegisteredFromTFStatusWithGIL(TF_Status* status) {
  TF_Code code = TF_GetCode(status);
  if (code != TF_OK) {
    // Acquire GIL for throwing exception.
    pybind11::gil_scoped_acquire acquire;
    SetRegisteredErrFromTFStatus(status);
    throw pybind11::error_already_set();
  }
}

}  // namespace tsl

namespace tensorflow {

using tsl::MaybeRaiseFromStatus;
using tsl::MaybeRaiseFromTFStatus;
using tsl::MaybeRaiseRegisteredFromStatus;
using tsl::MaybeRaiseRegisteredFromStatusWithGIL;
using tsl::MaybeRaiseRegisteredFromTFStatus;
using tsl::MaybeRaiseRegisteredFromTFStatusWithGIL;
using tsl::SetRegisteredErrFromStatus;
using tsl::SetRegisteredErrFromTFStatus;
}  // namespace tensorflow

namespace pybind11 {
namespace detail {

// Convert tensorflow::Status
//
// Raise an exception if a given status is not OK, otherwise return None.
//
// The correspondence between status codes and exception classes is given
// by PyExceptionRegistry. Note that the registry should be initialized
// in order to be used, see PyExceptionRegistry::Init.
template <>
struct type_caster<tensorflow::Status> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::Status, _("Status"));
  static handle cast(tensorflow::Status status, return_value_policy, handle) {
    tensorflow::MaybeRaiseFromStatus(status);
    return none().inc_ref();
  }
};

// Convert tensorflow::StatusOr
//
// Uses the same logic as the Abseil implementation: raise an exception if the
// status is not OK, otherwise return its payload.
template <typename PayloadType>
struct type_caster<tensorflow::StatusOr<PayloadType>> {
 public:
  using PayloadCaster = make_caster<PayloadType>;
  using StatusCaster = make_caster<tensorflow::Status>;
  static constexpr auto name = PayloadCaster::name;

  static handle cast(const tensorflow::StatusOr<PayloadType>* src,
                     return_value_policy policy, handle parent) {
    if (!src) return none().release();
    return cast_impl(*src, policy, parent);
  }

  static handle cast(const tensorflow::StatusOr<PayloadType>& src,
                     return_value_policy policy, handle parent) {
    return cast_impl(src, policy, parent);
  }

  static handle cast(tensorflow::StatusOr<PayloadType>&& src,
                     return_value_policy policy, handle parent) {
    return cast_impl(std::move(src), policy, parent);
  }

 private:
  template <typename CType>
  static handle cast_impl(CType&& src, return_value_policy policy,
                          handle parent) {
    if (src.ok()) {
      // Convert and return the payload.
      return PayloadCaster::cast(std::forward<CType>(src).value(), policy,
                                 parent);
    } else {
      // Convert and return the error.
      return StatusCaster::cast(std::forward<CType>(src).status(),
                                return_value_policy::move, parent);
    }
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_
