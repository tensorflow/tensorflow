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

#include "tensorflow/compiler/xla/python/traceback_manager.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

namespace py = pybind11;

}  // namespace

struct TracebackImpl {
  ~TracebackImpl();

  std::vector<TracebackManager::Frame> frames;

  // Computes a traceback for the current Python thread. Requires the GIL.
  bool GetTracebackForCurrentThread();
  bool operator==(const TracebackImpl& other) const;
  std::string ToString() const;
};

// We want Traceback objects to be safe to destroy without holding the GIL, so
// we defer destruction of the strings.
TracebackImpl::~TracebackImpl() {
  std::vector<py::object> objects;
  objects.reserve(2 * frames.size());
  for (TracebackManager::Frame& frame : frames) {
    objects.push_back(std::move(frame.file_name));
    objects.push_back(std::move(frame.function_name));
  }
  GlobalPyRefManager()->AddGarbage(absl::MakeSpan(objects));
}

bool TracebackImpl::operator==(const TracebackImpl& other) const {
  if (frames.size() != other.frames.size()) {
    return false;
  }
  for (int i = 0; i < frames.size(); ++i) {
    // Python strings are compared using pointer equality. This is cheap and
    // does not require calling back into the Python interpreter, but may mean
    // we miss some opportunities for deduplication of TracebackImpl objects.
    // However, we expect that function and file names are drawn from a fixed
    // pool of constants.
    if (frames[i].file_name.ptr() != other.frames[i].file_name.ptr() ||
        frames[i].function_name.ptr() != other.frames[i].function_name.ptr() ||
        frames[i].line_num != other.frames[i].line_num ||
        frames[i].function_start_line != other.frames[i].function_start_line) {
      return false;
    }
  }
  return true;
}

template <typename H>
H AbslHashValue(H h, const TracebackImpl& tb) {
  for (const TracebackManager::Frame& frame : tb.frames) {
    h = H::combine(std::move(h), frame.file_name.ptr(),
                   frame.function_name.ptr(), frame.line_num);
  }
  return h;
}
bool TracebackImpl::GetTracebackForCurrentThread() {
  PyThreadState* thread_state = PyGILState_GetThisThreadState();
  if (!thread_state) {
    return false;
  }
  frames.reserve(32);
  for (PyFrameObject* py_frame = thread_state->frame; py_frame != nullptr;
       py_frame = py_frame->f_back) {
    frames.resize(frames.size() + 1);
    TracebackManager::Frame& frame = frames.back();
    PyCodeObject* code = py_frame->f_code;
    if (!code) {
      return false;
    }
    frame.line_num = PyFrame_GetLineNumber(py_frame);
    frame.file_name = py::str(code->co_filename);
    frame.function_name = py::str(code->co_name);
    frame.function_start_line = code->co_firstlineno;
  }
  return true;
}

std::string TracebackImpl::ToString() const {
  std::vector<std::string> frame_strs;
  frame_strs.reserve(frames.size());
  for (const TracebackManager::Frame& frame : frames) {
    frame_strs.push_back(absl::StrFormat("%s:%d (%s)", frame.file_name,
                                         frame.line_num, frame.function_name));
  }
  return absl::StrJoin(frame_strs, "\n");
}

TracebackManager::Traceback::Traceback(
    TracebackManager* manager, std::pair<TracebackImpl const, int>* impl)
    : manager_(manager), impl_(impl) {
  DCHECK(manager_);
  ++impl->second;
}

TracebackManager::Traceback::~Traceback() {
  if (manager_) {
    --impl_->second;
    if (impl_->second == 0) {
      manager_->tracebacks_.erase(impl_->first);
    }
  }
}

TracebackManager::Traceback::Traceback(const Traceback& other)
    : manager_(other.manager_), impl_(other.impl_) {
  if (manager_) {
    ++impl_->second;
  }
}

TracebackManager::Traceback::Traceback(Traceback&& other)
    : manager_(other.manager_), impl_(other.impl_) {
  other.manager_ = nullptr;
  other.impl_ = nullptr;
}

TracebackManager::Traceback& TracebackManager::Traceback::operator=(
    const TracebackManager::Traceback& other) {
  manager_ = other.manager_;
  impl_ = other.impl_;
  if (manager_) {
    ++impl_->second;
  }
  return *this;
}

TracebackManager::Traceback& TracebackManager::Traceback::operator=(
    TracebackManager::Traceback&& other) {
  std::swap(manager_, other.manager_);
  std::swap(impl_, other.impl_);
  return *this;
}

std::string TracebackManager::Traceback::ToString() const {
  // We require the GIL because we manipulate Python strings.
  CHECK(PyGILState_Check());
  if (!manager_) {
    // Don't crash if called on a default-constructed Traceback.
    return "<unknown>";
  }
  return impl_->first.ToString();
}

const std::vector<TracebackManager::Frame>*
TracebackManager::Traceback::Frames() const {
  return &impl_->first.frames;
}

/*static*/ TracebackManager* TracebackManager::Get() {
  static TracebackManager* manager = new TracebackManager;
  return manager;
}

TracebackManager::TracebackManager() = default;
TracebackManager::~TracebackManager() = default;

absl::optional<TracebackManager::Traceback> TracebackManager::GetTraceback() {
  if (!enabled_) {
    return absl::nullopt;
  }
  CHECK(PyGILState_Check());
  TracebackImpl impl;
  if (!impl.GetTracebackForCurrentThread()) {
    return absl::nullopt;
  }
  auto it = tracebacks_.emplace(impl, 0);
  return Traceback(this, &*it.first);
}

void TracebackManager::SetEnabled(bool enabled) { enabled_ = enabled; }

}  // namespace xla
