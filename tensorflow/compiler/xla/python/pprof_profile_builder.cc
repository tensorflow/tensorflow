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

#include "tensorflow/compiler/xla/python/pprof_profile_builder.h"

#include <string>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

namespace py = pybind11;

PprofProfileBuilder::PprofProfileBuilder() { CHECK_EQ(0, StringId("")); }

int PprofProfileBuilder::StringId(const std::string& s) {
  auto ret = strings_.emplace(s, profile_.string_table_size());
  if (ret.second) {
    profile_.add_string_table(s);
  }
  return ret.first->second;
}

int PprofProfileBuilder::FunctionId(PyCodeObject* code) {
  // +1 because id 0 is reserved.
  auto ret = functions_.emplace(code, profile_.function_size() + 1);
  if (ret.second) {
    auto* function = profile_.add_function();
    function->set_id(ret.first->second);
    int name = StringId(py::str(code->co_name));
    function->set_name(name);
    function->set_system_name(name);
    function->set_filename(StringId(py::str(code->co_filename)));
    function->set_start_line(code->co_firstlineno);
  }
  return ret.first->second;
}

int PprofProfileBuilder::LocationId(PyCodeObject* code, int instruction) {
  // +1 because id 0 is reserved.
  auto ret = locations_.emplace(std::make_pair(code, instruction),
                                profile_.location_size() + 1);
  if (ret.second) {
    auto* location = profile_.add_location();
    location->set_id(ret.first->second);
    auto* line = location->add_line();
    line->set_function_id(FunctionId(code));
    line->set_line(PyCode_Addr2Line(code, instruction));
  }
  return ret.first->second;
}

StatusOr<pybind11::bytes> JsonToPprofProfile(std::string json) {
  tensorflow::tfprof::pprof::Profile profile;
  auto status = tensorflow::protobuf::util::JsonStringToMessage(json, &profile);
  if (!status.ok()) {
    // TODO(phawkins): the explicit `std::string` cast here is to work around
    // https://github.com/google/jax/issues/9534 which appears to be an ABSL and
    // protobuf version compatibility problem.
    return InvalidArgument("JSON parsing failed: %s",
                           std::string{status.message()});
  }
  return py::bytes(profile.SerializeAsString());
}

StatusOr<std::string> PprofProfileToJson(py::bytes binary_proto) {
  tensorflow::tfprof::pprof::Profile profile;
  profile.ParseFromString(binary_proto);
  std::string output;
  auto status =
      tensorflow::protobuf::util::MessageToJsonString(profile, &output);
  if (!status.ok()) {
    // TODO(phawkins): the explicit `std::string` cast here is to work around
    // https://github.com/google/jax/issues/9534 which appears to be an ABSL and
    // protobuf version compatibility problem.
    return InvalidArgument("JSON printing failed: %s",
                           std::string{status.message()});
  }
  return output;
}

}  // namespace xla
