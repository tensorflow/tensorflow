/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/traceback.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace py = pybind11;

bool Traceback::enabled_ = true;

Traceback::~Traceback() {
  // We want Traceback objects to be safe to destroy without holding the
  // GIL, so we defer destruction of the strings.
  GlobalPyRefManager()->AddGarbage(frames_);
}

std::string Traceback::Frame::ToString() const {
  return absl::StrFormat("%s:%d (%s)", file_name, line_num, function_name);
}

std::string Traceback::ToString() const {
  std::vector<std::string> frame_strs;
  frame_strs.reserve(frames_.size());
  for (const Frame& frame : Frames()) {
    frame_strs.push_back(frame.ToString());
  }
  return absl::StrJoin(frame_strs, "\n");
}

std::vector<Traceback::Frame> Traceback::Frames() const {
  // We require the GIL because we manipulate Python strings.
  CHECK(PyGILState_Check());
  std::vector<Traceback::Frame> frames;
  frames.reserve(frames_.size());
  for (const auto& frame : frames_) {
    frames.push_back(Frame{
        std::string(py::reinterpret_borrow<py::str>(frame.first->co_filename)),
        std::string(py::reinterpret_borrow<py::str>(frame.first->co_name)),
        frame.first->co_firstlineno,
        PyCode_Addr2Line(frame.first, frame.second)});
  }
  return frames;
}

std::shared_ptr<Traceback> Traceback::Get() {
  DCHECK(PyGILState_Check());
  if (!enabled_) {
    return nullptr;
  }
  auto tb = std::make_shared<Traceback>();
  const PyThreadState* thread_state = PyThreadState_GET();
  for (PyFrameObject* py_frame = thread_state->frame; py_frame != nullptr;
       py_frame = py_frame->f_back) {
    Py_INCREF(py_frame->f_code);
    tb->frames_.emplace_back(py_frame->f_code, py_frame->f_lasti);
  }
  return tb;
}

void Traceback::SetEnabled(bool enabled) { enabled_ = enabled; }

}  // namespace xla
