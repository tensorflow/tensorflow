/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/python/profiler/xplane_visitor.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // IWYU pragma: keep. For automatic conversion of std::string to Python string.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace tensorflow::profiler::python {

namespace nb = nanobind;
using tensorflow::profiler::XEvent;
using tensorflow::profiler::XLine;
using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tensorflow::profiler::XStat;

// Converts a XStat object to a Python tuple. For compatibility reasons, we
// always return of the same sizes.
nb::tuple stats_to_tuple(const XStat& stat, const XPlane* plane) {
  if (plane->stat_metadata().contains(stat.metadata_id())) {
    const std::string& name =
        plane->stat_metadata().at(stat.metadata_id()).name();
    switch (stat.value_case()) {
      case XStat::kDoubleValue:
        return nb::make_tuple(name, nb::cast(stat.double_value()));
        break;
      case XStat::kUint64Value:
        return nb::make_tuple(name, nb::cast(stat.uint64_value()));
        break;
      case XStat::kInt64Value:
        return nb::make_tuple(name, nb::cast(stat.int64_value()));
        break;
      case XStat::kStrValue:
        return nb::make_tuple(name, stat.str_value());
        break;
      case XStat::kBytesValue:
        return nb::make_tuple(name, stat.bytes_value());
        break;
      case XStat::kRefValue:
        if (plane->stat_metadata().contains(stat.ref_value())) {
          return nb::make_tuple(
              name, plane->stat_metadata().at(stat.ref_value()).name());
        } else {
          return nb::make_tuple(name, "");
        }
        break;
      default:
        LOG(ERROR) << "Unsupported stat value type: " << stat.value_case();
        break;
    }
  }
  return nb::make_tuple(nb::none(), nb::none());
}

XEventVisitor::XEventVisitor(const XEvent* event, int64_t line_timestamp_ns,
                             const XPlane* plane,
                             std::shared_ptr<const XSpace> xspace)
    : event_(event),
      plane_(plane),
      line_timestamp_ns_(line_timestamp_ns),
      xspace_(xspace) {
  CHECK_NOTNULL(event_);
  CHECK_NOTNULL(plane_);
  CHECK_NOTNULL(xspace_);
}

double XEventVisitor::start_ns() const {
  return event_->offset_ps() / 1000 + line_timestamp_ns_;
}

double XEventVisitor::duration_ns() const {
  return event_->duration_ps() / 1000;
}

double XEventVisitor::end_ns() const { return start_ns() + duration_ns(); }

std::string XEventVisitor::name() const {
  if (plane_->event_metadata().contains(event_->metadata_id())) {
    return plane_->event_metadata().at(event_->metadata_id()).name();
  }
  return "";
}

VisitorIterator<nb::tuple, XStat> XEventVisitor::stats_begin() {
  return VisitorIterator<nb::tuple, XStat>(
      &event_->stats(),
      [this](const XStat& stat) { return stats_to_tuple(stat, plane_); });
}
VisitorIterator<nb::tuple, XStat> XEventVisitor::stats_end() {
  return VisitorIterator<nb::tuple, XStat>(
      &event_->stats(),
      [this](const XStat& stat) { return stats_to_tuple(stat, plane_); },
      event_->stats().size());
}

XLineVisitor::XLineVisitor(const XLine* line, const XPlane* plane,
                           std::shared_ptr<const XSpace> xspace)
    : line_(line), plane_(plane), xspace_(xspace) {
  CHECK_NOTNULL(line_);
  CHECK_NOTNULL(plane_);
  CHECK_NOTNULL(xspace_);
}

const std::string& XLineVisitor::name() const { return line_->name(); }

VisitorIterator<XEventVisitor, XEvent> XLineVisitor::events_begin() {
  return VisitorIterator<XEventVisitor, XEvent>(
      &line_->events(), [this](const XEvent& event) {
        return XEventVisitor(&event, line_->timestamp_ns(), plane_, xspace_);
      });
}

VisitorIterator<XEventVisitor, XEvent> XLineVisitor::events_end() {
  return VisitorIterator<XEventVisitor, XEvent>(
      &line_->events(),
      [this](const XEvent& event) {
        return XEventVisitor(&event, line_->timestamp_ns(), plane_, xspace_);
      },
      line_->events().size());
}

XPlaneVisitor::XPlaneVisitor(const XPlane* plane,
                             std::shared_ptr<const XSpace> xspace)
    : plane_(plane), xspace_(xspace) {
  CHECK_NOTNULL(plane_);
  CHECK_NOTNULL(xspace_);
}

const std::string& XPlaneVisitor::name() const { return plane_->name(); }

VisitorIterator<XLineVisitor, XLine> XPlaneVisitor::lines_begin() {
  return VisitorIterator<XLineVisitor, XLine>(
      &plane_->lines(), [this](const XLine& line) {
        return XLineVisitor(&line, plane_, xspace_);
      });
}
VisitorIterator<XLineVisitor, XLine> XPlaneVisitor::lines_end() {
  return VisitorIterator<XLineVisitor, XLine>(
      &plane_->lines(),
      [this](const XLine& line) {
        return XLineVisitor(&line, plane_, xspace_);
      },
      plane_->lines().size());
}

VisitorIterator<nb::tuple, XStat> XPlaneVisitor::stats_begin() {
  return VisitorIterator<nb::tuple, XStat>(
      &plane_->stats(),
      [this](const XStat& stat) { return stats_to_tuple(stat, plane_); });
}

VisitorIterator<nb::tuple, XStat> XPlaneVisitor::stats_end() {
  return VisitorIterator<nb::tuple, XStat>(
      &plane_->stats(),
      [this](const XStat& stat) { return stats_to_tuple(stat, plane_); },
      plane_->stats().size());
}

/*static*/ XSpaceVisitor XSpaceVisitor::from_serialized_xspace(
    const nb::bytes& serialized_xspace) {
  return XSpaceVisitor(serialized_xspace);
}

/*static*/ XSpaceVisitor XSpaceVisitor::from_file(
    const std::string& proto_file_path) {
  std::string serialized_xspace;
  TF_CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), proto_file_path,
                                    &serialized_xspace));
  return XSpaceVisitor(serialized_xspace.c_str(), serialized_xspace.size());
}

/*static*/ XSpaceVisitor XSpaceVisitor::from_raw_cpp_ptr(nb::capsule capsule) {
  auto raw_ptr = static_cast<XSpace*>(capsule.data());
  auto proto_ptr = std::shared_ptr<XSpace>(raw_ptr);

  return XSpaceVisitor(proto_ptr);
}

XSpaceVisitor::XSpaceVisitor(const char* serialized_xspace_ptr,
                             size_t serialized_xspace_size) {
  CHECK_NOTNULL(serialized_xspace_ptr);

  if (!xspace_) {
    xspace_ = std::make_shared<XSpace>();
  }
  CHECK(xspace_->ParseFromArray(serialized_xspace_ptr, serialized_xspace_size));
}

/*explicit*/ XSpaceVisitor::XSpaceVisitor(std::shared_ptr<XSpace> xspace_ptr) {
  xspace_ = xspace_ptr;
}

/*explicit*/ XSpaceVisitor::XSpaceVisitor(const nb::bytes& serialized_xspace) {
  if (!xspace_) {
    xspace_ = std::make_shared<XSpace>();
  }
  CHECK(xspace_->ParseFromArray(serialized_xspace.data(),
                                serialized_xspace.size()));
}

VisitorIterator<XPlaneVisitor, XPlane> XSpaceVisitor::planes_begin() {
  return VisitorIterator<XPlaneVisitor, XPlane>(
      &xspace_->planes(),
      [this](const XPlane& plane) { return XPlaneVisitor(&plane, xspace_); });
}

VisitorIterator<XPlaneVisitor, XPlane> XSpaceVisitor::planes_end() {
  return VisitorIterator<XPlaneVisitor, XPlane>(
      &xspace_->planes(),
      [this](const XPlane& plane) { return XPlaneVisitor(&plane, xspace_); },
      xspace_->planes().size());
}

XPlaneVisitor* XSpaceVisitor::find_plane_with_name(
    const std::string& name) const {
  for (const auto& plane : xspace_->planes()) {
    if (plane.name() == name) {
      return new XPlaneVisitor(&plane, xspace_);
    }
  }
  return nullptr;
}

}  // namespace tensorflow::profiler::python
