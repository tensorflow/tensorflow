/* Copyright 2025 The OpenXLA Authors.

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
#include <string>

#include "nanobind/make_iterator.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/python/profiler/profile_data_lib.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {

namespace nb = nanobind;
namespace profpy = tensorflow::profiler::python;

NB_MODULE(_profile_data, m) {
  nb::class_<profpy::ProfileEvent>(m, "ProfileEvent")
      .def_prop_ro("start_ns", &profpy::ProfileEvent::start_ns)
      .def_prop_ro("duration_ns", &profpy::ProfileEvent::duration_ns)
      .def_prop_ro("end_ns", &profpy::ProfileEvent::end_ns)
      .def_prop_ro("name", &profpy::ProfileEvent::name)
      .def_prop_ro(
          "stats",
          [](profpy::ProfileEvent&& e) {
            return nb::make_iterator(nb::type<nb::tuple>(), "event_stats",
                                     e.stats_begin(), e.stats_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<profpy::ProfileLine>(m, "ProfileLine")
      .def_prop_ro("name", &profpy::ProfileLine::name)
      .def_prop_ro(
          "events",
          [](profpy::ProfileLine&& l) {
            return nb::make_iterator(nb::type<profpy::ProfileEvent>(), "events",
                                     l.events_begin(), l.events_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<profpy::ProfilePlane>(m, "ProfilePlane")
      .def_prop_ro("name", &profpy::ProfilePlane::name)
      .def_prop_ro(
          "lines",
          [](profpy::ProfilePlane&& p) {
            return nb::make_iterator(nb::type<profpy::ProfileLine>(), "lines",
                                     p.lines_begin(), p.lines_end());
          },
          nb::keep_alive<0, 1>())
      .def_prop_ro(
          "stats",
          [](profpy::ProfilePlane&& p) {
            return nb::make_iterator(nb::type<nb::tuple>(), "plane_stats",
                                     p.stats_begin(), p.stats_end());
          },
          nb::keep_alive<0, 1>());
  nb::class_<profpy::ProfileData>(m, "ProfileData")
      .def_static("from_raw_cpp_ptr", &profpy::ProfileData::from_raw_cpp_ptr,
                  nb::arg("capsule"))
      .def_static("from_file", &profpy::ProfileData::from_file,
                  nb::arg("proto_file_path"),
                  "Creates a ProfileData from a serialized XSpace proto file.")
      .def_static("from_serialized_xspace",
                  &profpy::ProfileData::from_serialized_xspace,
                  nb::arg("serialized_xspace"))
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
      .def("find_plane_with_name", &profpy::ProfileData::find_plane_with_name,
           nb::arg("name"), nb::keep_alive<0, 1>())
      .def_prop_ro(
          "planes",
          [](profpy::ProfileData&& s) {
            return nb::make_iterator(nb::type<profpy::ProfilePlane>(), "planes",
                                     s.planes_begin(), s.planes_end());
          },
          nb::keep_alive<0, 1>());
}

}  // namespace xla
