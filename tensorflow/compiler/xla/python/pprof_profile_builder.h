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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PPROF_PROFILE_BUILDER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PPROF_PROFILE_BUILDER_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/profiler/protobuf/profile.pb.h"

namespace xla {

// Helper class for building pprof::Profile profiles.
class PprofProfileBuilder {
 public:
  PprofProfileBuilder();
  tensorflow::tfprof::pprof::Profile& profile() { return profile_; }

  // Adds or returns the ID of `s` in the table.
  int StringId(const std::string& s);

  // Adds or returns the ID of a function.
  int FunctionId(PyCodeObject* code);

  // Adds or returns the ID of a code location.
  int LocationId(PyCodeObject* code, int instruction);

 private:
  tensorflow::tfprof::pprof::Profile profile_;

  absl::flat_hash_map<std::string, int> strings_;
  absl::flat_hash_map<PyCodeObject*, int> functions_;
  absl::flat_hash_map<std::pair<PyCodeObject*, int>, int> locations_;
};

// Converts the JSON representation of a pprof profile protocol buffer into
// a serialized protocol buffer. We want to allow Python code to construct pprof
// protocol buffers, but we don't want to export the generated protocol buffer
// bindings for Python because they cause conflicts between multiple Python
// extensions that contain the same protocol buffer message. Instead, we accept
// a JSON representation from Python and use this function to serialize it to
// a uncompressed binary protocol buffer.
StatusOr<pybind11::bytes> JsonToPprofProfile(std::string json);

// The reverse, useful for testing.
StatusOr<std::string> PprofProfileToJson(pybind11::bytes binary_proto);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PPROF_PROFILE_BUILDER_H_
