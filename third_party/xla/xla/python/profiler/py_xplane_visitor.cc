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

#include <nanobind/make_iterator.h>  // For automatic conversion of std::iterator to Python iterable.
#include <nanobind/stl/string.h>  // For automatic conversion of std::string to Python string.

#include "xla/python/profiler/xplane_visitor.h"

namespace {

namespace nb = nanobind;
// NOLINTBEGIN(build/namespaces)
using namespace nb::literals;
using namespace tensorflow::profiler::python;
// NOLINTEND(build/namespaces)

NB_MODULE(xplane_visitor, m) {
  nb::class_<XEventVisitor>(m, "XEventVisitor")
      .def_prop_ro("start_ns", &XEventVisitor::start_ns)
      .def_prop_ro("duration_ns", &XEventVisitor::duration_ns)
      .def_prop_ro("end_ns", &XEventVisitor::end_ns)
      .def_prop_ro("name", &XEventVisitor::name)
      .def_prop_ro(
          "stats",
          [](XEventVisitor&& e) {
            return nb::make_iterator(nb::type<nb::tuple>(), "event_stats",
                                     e.stats_begin(), e.stats_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<XLineVisitor>(m, "XLineVisitor")
      .def_prop_ro("name", &XLineVisitor::name)
      .def_prop_ro(
          "events",
          [](XLineVisitor&& l) {
            return nb::make_iterator(nb::type<XEventVisitor>(), "events",
                                     l.events_begin(), l.events_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<XPlaneVisitor>(m, "XPlaneVisitor")
      .def_prop_ro("name", &XPlaneVisitor::name)
      .def_prop_ro(
          "lines",
          [](XPlaneVisitor&& p) {
            return nb::make_iterator(nb::type<XLineVisitor>(), "lines",
                                     p.lines_begin(), p.lines_end());
          },
          nb::keep_alive<0, 1>())
      .def_prop_ro(
          "stats",
          [](XPlaneVisitor&& p) {
            return nb::make_iterator(nb::type<nb::tuple>(), "plane_stats",
                                     p.stats_begin(), p.stats_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<XSpaceVisitor>(m, "XSpaceVisitor")
      .def_static("from_raw_cpp_ptr", &XSpaceVisitor::from_raw_cpp_ptr,
                  "capsule"_a)
      .def_static(
          "from_file", &XSpaceVisitor::from_file, "proto_file_path"_a,
          "Creates an XSpaceVisitor from a serialized XSpace proto file.")
      .def_static("from_serialized_xspace",
                  &XSpaceVisitor::from_serialized_xspace, "serialized_xspace"_a)
      .def(nb::init<const nb::bytes&>())
      .def("find_plane_with_name", &XSpaceVisitor::find_plane_with_name,
           "name"_a, nb::keep_alive<0, 1>())
      .def_prop_ro(
          "planes",
          [](XSpaceVisitor&& s) {
            return nb::make_iterator(nb::type<XPlaneVisitor>(), "planes",
                                     s.planes_begin(), s.planes_end());
          },
          nb::keep_alive<0, 1>());
}

}  // namespace
