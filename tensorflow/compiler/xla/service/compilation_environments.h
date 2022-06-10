/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_SERVICE_XLA_COMPILATION_ENVIRONMENTS_H_
#define TENSORFLOW_COMPILER_SERVICE_XLA_COMPILATION_ENVIRONMENTS_H_

#include <memory>
#include <typeindex>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

// A class for holding CompilationEnvironments, i.e., protos holding the values
// of command line flags and environment variables that affect compilation.
//
// CompilationEnvironments uses lazy initialization, (see GetEnv() for more
// details). Lazy initialization is used so we can avoid:
// A) Requiring every code path to explitily construct all needed compilation
//    environments, particularly when the default constructed environment is
//    all we need AND
// B) Requiring CompilationEnvironments to implicitly construct all needed
//    environments, thereby requiring it to statically know the types of all
//    such environments
//
// CompilationEnvironments is not thread-safe.
class CompilationEnvironments {
 public:
  CompilationEnvironments() = default;
  CompilationEnvironments(const CompilationEnvironments& rhs) { *this = rhs; }
  CompilationEnvironments& operator=(const CompilationEnvironments& rhs);
  ~CompilationEnvironments() = default;

  // Users of CompilationEnvironments must specialize this method for each type
  // of CompilationEnvironment they wish to use in code.
  //
  // T must be a type of proto message.
  template <typename T>
  static std::unique_ptr<T> CreateDefaultEnv() = delete;

  // Adds env to the list of CompilationEnvironments. If an environment with
  // std::type_index equal to env.GetTypeid() has already been added, env
  // will replace it.
  void AddEnv(std::unique_ptr<tensorflow::protobuf::Message> env);

  // Returns the CompilationEnvironment corresponding to T. If such an
  // environment has not been added, CreateDefaultEnv<T>() will be called to
  // create one that is then added.
  //
  // GetEnv() is not const because it can perform lazy initialization, thereby
  // modifying the CompilationEnvironments's data members.
  //
  // GetEnv<T> will not compile for type T, unless CreateDefaultEnv<T> is
  // defined.
  template <typename T>
  const T& GetEnv();

  // Removes all added environments.
  void Clear() { environments_.clear(); }

 private:
  absl::flat_hash_map<const tensorflow::protobuf::Descriptor*,
                      std::unique_ptr<tensorflow::protobuf::Message>>
      environments_;
};

// ----- Template implementation below -----

// Make sure no one tries to specialize CreateDefaultEnv() for raw
// tensorflow::protobuf::Message. Specialization should always be for a specific
// type of proto message.
template <>
std::unique_ptr<tensorflow::protobuf::Message>
CompilationEnvironments::CreateDefaultEnv() = delete;

template <typename T>
const T& CompilationEnvironments::GetEnv() {
  auto descriptor = T::descriptor();
  auto it = environments_.find(descriptor);
  if (it == environments_.end()) {
    AddEnv(CreateDefaultEnv<T>());
    it = environments_.find(descriptor);
  }
  return tensorflow::down_cast<const T&>(*it->second);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_SERVICE_XLA_COMPILATION_ENVIRONMENTS_H_
