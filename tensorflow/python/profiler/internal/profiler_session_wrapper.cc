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

#include <memory>

#include "include/pybind11/pybind11.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

namespace {

class ProfilerSessionWrapper {
 public:
  void Start() {
    session_ = tensorflow::ProfilerSession::Create();
    tensorflow::MaybeRaiseRegisteredFromStatus(session_->Status());
  }

  py::bytes Stop() {
    tensorflow::string content;
    if (session_ != nullptr) {
      tensorflow::Status status = session_->SerializeToString(&content);
      tensorflow::MaybeRaiseRegisteredFromStatus(status);
      session_.reset();
    }
    // The content is not valid UTF-8, so it must be converted to bytes.
    return py::bytes(content);
  }

 private:
  std::unique_ptr<tensorflow::ProfilerSession> session_;
};

}  // namespace

PYBIND11_MODULE(_pywrap_profiler_session, m) {
  py::class_<ProfilerSessionWrapper> profiler_session_class(m,
                                                            "ProfilerSession");
  profiler_session_class.def(py::init<>())
      .def("start", &ProfilerSessionWrapper::Start)
      .def("stop", &ProfilerSessionWrapper::Stop);
};
