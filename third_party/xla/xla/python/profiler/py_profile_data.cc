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

#include <memory>
#include <string>

#include "xla/python/profiler/profile_data.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {

namespace nb = nanobind;
// NOLINTBEGIN(build/namespaces)
using namespace nb::literals;
using namespace tensorflow::profiler::python;
// NOLINTEND(build/namespaces)

NB_MODULE(profile_data, m) {
  nb::class_<ProfileEvent>(m, "ProfileEvent")
      .def_prop_ro("start_ns", &ProfileEvent::start_ns)
      .def_prop_ro("duration_ns", &ProfileEvent::duration_ns)
      .def_prop_ro("end_ns", &ProfileEvent::end_ns)
      .def_prop_ro("name", &ProfileEvent::name)
      .def_prop_ro(
          "stats",
          [](ProfileEvent&& e) {
            return nb::make_iterator(nb::type<nb::tuple>(), "event_stats",
                                     e.stats_begin(), e.stats_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<ProfileLine>(m, "ProfileLine")
      .def_prop_ro("name", &ProfileLine::name)
      .def_prop_ro(
          "events",
          [](ProfileLine&& l) {
            return nb::make_iterator(nb::type<ProfileEvent>(), "events",
                                     l.events_begin(), l.events_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<ProfilePlane>(m, "ProfilePlane")
      .def_prop_ro("name", &ProfilePlane::name)
      .def_prop_ro(
          "lines",
          [](ProfilePlane&& p) {
            return nb::make_iterator(nb::type<ProfileLine>(), "lines",
                                     p.lines_begin(), p.lines_end());
          },
          nb::keep_alive<0, 1>())
      .def_prop_ro(
          "stats",
          [](ProfilePlane&& p) {
            return nb::make_iterator(nb::type<nb::tuple>(), "plane_stats",
                                     p.stats_begin(), p.stats_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<ProfileData>(m, "ProfileData")
      .def_static("from_raw_cpp_ptr", &ProfileData::from_raw_cpp_ptr,
                  "capsule"_a)
      .def_static("from_file", &ProfileData::from_file, "proto_file_path"_a,
                  "Creates a ProfileData from a serialized XSpace proto file.")
      .def_static("from_serialized_xspace",
                  &ProfileData::from_serialized_xspace, "serialized_xspace"_a)
      .def_static("from_text_proto",
                  [](const std::string& text_proto) {
                    auto xspace =
                        std::make_shared<tensorflow::profiler::XSpace>();
                    tsl::protobuf::TextFormat::ParseFromString(text_proto,
                                                               xspace.get());
                    return tensorflow::profiler::python::ProfileData(xspace);
                  })
      .def_static("text_proto_to_serialized_xspace",
                  [](const std::string& text_proto) {
                    tensorflow::profiler::XSpace xspace;
                    tsl::protobuf::TextFormat::ParseFromString(text_proto,
                                                               &xspace);
                    const auto serialized = xspace.SerializeAsString();
                    return nb::bytes(serialized.data(), serialized.size());
                  })
      .def(nb::init<const nb::bytes&>())
      .def("find_plane_with_name", &ProfileData::find_plane_with_name, "name"_a,
           nb::keep_alive<0, 1>())
      .def_prop_ro(
          "planes",
          [](ProfileData&& s) {
            return nb::make_iterator(nb::type<ProfilePlane>(), "planes",
                                     s.planes_begin(), s.planes_end());
          },
          nb::keep_alive<0, 1>());
}

}  // namespace
