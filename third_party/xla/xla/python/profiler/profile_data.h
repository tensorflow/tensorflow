/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PROFILER_PROFILE_DATA_H_
#define XLA_PYTHON_PROFILER_PROFILE_DATA_H_

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow::profiler::python {

namespace nb = nanobind;

// A simple iterator that converts a proto repeated field to a Python iterable
// with a customized conversion function.
template <typename OutputType, typename InputType>
class VisitorIterator
    : public std::iterator<std::input_iterator_tag, OutputType> {
 public:
  VisitorIterator(
      const tsl::protobuf::RepeatedPtrField<InputType>* values,
      const std::function<OutputType(const InputType&)>& make_visitor,
      int pos = 0)
      : values_(values), make_visitor_(make_visitor), pos_(pos) {
    CHECK_NOTNULL(values_);
    CHECK_GE(pos_, 0);
    CHECK_LE(pos_, values_->size());
  }

  // Prefix increment operator.
  VisitorIterator& operator++() {
    ++pos_;
    return *this;
  }

  // Postfix increment operator.
  VisitorIterator operator++(int) {
    VisitorIterator tmp(*this);
    operator++();
    return tmp;
  }

  bool operator==(const VisitorIterator& rhs) const {
    return pos_ == rhs.pos_ && values_ == rhs.values_;
  }

  bool operator!=(const VisitorIterator& rhs) const {
    return pos_ != rhs.pos_ || values_ != rhs.values_;
  }

  OutputType operator*() { return make_visitor_((*values_)[pos_]); }

 private:
  const tsl::protobuf::RepeatedPtrField<InputType>* values_;
  const std::function<OutputType(const InputType&)> make_visitor_;
  int pos_ = 0;
};

class ProfileEvent {
 public:
  ProfileEvent() = delete;

  ProfileEvent(const tensorflow::profiler::XEvent* event,
               int64_t line_timestamp_ns,
               const tensorflow::profiler::XPlane* plane,
               std::shared_ptr<const tensorflow::profiler::XSpace> xspace);

  double start_ns() const;

  double duration_ns() const;

  double end_ns() const;

  std::string name() const;

  VisitorIterator<nb::tuple, tensorflow::profiler::XStat> stats_begin();
  VisitorIterator<nb::tuple, tensorflow::profiler::XStat> stats_end();

 private:
  const XEvent* event_;
  const XPlane* plane_;
  const int64_t line_timestamp_ns_;
  // The actual XSpace protobuf we are wrapping around. A shared ptr is used so
  // the different levels of  visitors (ProfileData, ProfilePlane,
  // ProfileLine, etc.) don't depend on the lifetime of others.
  const std::shared_ptr<const XSpace> xspace_;
};

class ProfileLine {
 public:
  ProfileLine() = delete;

  ProfileLine(const tensorflow::profiler::XLine* line,
              const tensorflow::profiler::XPlane* plane,
              std::shared_ptr<const tensorflow::profiler::XSpace> xspace);

  const std::string& name() const;

  VisitorIterator<ProfileEvent, tensorflow::profiler::XEvent> events_begin();
  VisitorIterator<ProfileEvent, tensorflow::profiler::XEvent> events_end();

 private:
  const XLine* line_;
  const XPlane* plane_;
  // The actual XSpace protobuf we are wrapping around. A shared ptr is used so
  // the different levels of  visitors (ProfileData, ProfilePlane,
  // ProfileLine, etc.) don't depend on the lifetime of others.
  const std::shared_ptr<const XSpace> xspace_;
};

class ProfilePlane {
 public:
  ProfilePlane() = delete;

  ProfilePlane(const tensorflow::profiler::XPlane* plane,
               std::shared_ptr<const tensorflow::profiler::XSpace> xspace);

  const std::string& name() const;

  VisitorIterator<ProfileLine, tensorflow::profiler::XLine> lines_begin();
  VisitorIterator<ProfileLine, tensorflow::profiler::XLine> lines_end();

  VisitorIterator<nb::tuple, tensorflow::profiler::XStat> stats_begin();

  VisitorIterator<nb::tuple, tensorflow::profiler::XStat> stats_end();

 private:
  const XPlane* plane_;
  // The actual XSpace protobuf we are wrapping around. A shared ptr is used so
  // the different levels of  visitors (ProfileData, ProfilePlane,
  // ProfileLine, etc.) don't depend on the lifetime of others.
  const std::shared_ptr<const XSpace> xspace_;
};

class ProfileData {
 public:
  static ProfileData from_serialized_xspace(const nb::bytes& serialized_xspace);

  static ProfileData from_file(const std::string& proto_file_path);

  static ProfileData from_raw_cpp_ptr(nb::capsule capsule);

  ProfileData() = delete;

  ProfileData(const char* serialized_xspace_ptr, size_t serialized_xspace_size);

  explicit ProfileData(std::shared_ptr<XSpace> xspace_ptr);

  explicit ProfileData(const nb::bytes& serialized_xspace);

  VisitorIterator<ProfilePlane, XPlane> planes_begin();

  VisitorIterator<ProfilePlane, XPlane> planes_end();

  ProfilePlane* find_plane_with_name(const std::string& name) const;

 private:
  // The actual XSpace protobuf we are wrapping around. A shared ptr is used so
  // the different levels of  visitors (ProfileData, ProfilePlane,
  // ProfileLine, etc.) don't depend on the lifetime of others.
  std::shared_ptr<XSpace> xspace_;
};

ProfileData from_serialized_xspace(const std::string& serialized_xspace);

ProfileData from_file(const std::string& proto_file_path);

}  // namespace tensorflow::profiler::python

#endif  // XLA_PYTHON_PROFILER_PROFILE_DATA_H_
