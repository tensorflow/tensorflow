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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_ENVIRONMENTS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_ENVIRONMENTS_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <typeindex>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/tsl/platform/casts.h"
#include "tensorflow/tsl/platform/protobuf.h"

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

  // Whenever an environment is added to CompilationEnvironments, even when
  // GetEnv() adds a lazily initialized one, it is passed to this method. The
  // result of this method is the environment that is used by
  // CompilationEnvironments. This allows environment authors to do things like
  // populate missing fields in an added environment.
  //
  // Users of CompilationEnvironments must specialize this method for each type
  // of CompilationEnvironment they wish to use in code.
  //
  // The input env may be null.
  //
  // REQUIRES:
  // - T must be a type of proto message.
  // - The output is *not* allowed to be null, even for null input.
  template <typename T>
  static std::unique_ptr<T> ProcessNewEnv(std::unique_ptr<T> env) = delete;

  // Adds env to the list of CompilationEnvironments. If an environment with
  // std::type_index equal to env.GetTypeid() has already been added, env
  // will replace it.
  //
  // All added environments are processed via ProcessNewEnv().
  //
  // AddEnv<T> will not compile for type T, unless ProcessNewEnv<T> is defined.
  template <typename T>
  void AddEnv(std::unique_ptr<T> env);

  // Returns the CompilationEnvironment corresponding to T. If such an
  // environment has not been added, ProcessNewEnv<T>(nullptr) will be added
  // and returned.
  //
  // GetMutableEnv()/GetEnv() are not const because they can perform lazy
  // initialization, thereby modifying the CompilationEnvironments's data
  // members.
  //
  // GetMutableEnv<T>/GetEnv<T> will not compile for type T, unless
  // ProcessNewEnv<T> is defined.
  template <typename T>
  T& GetMutableEnv();
  template <typename T>
  const T& GetEnv();

  // Removes all added environments.
  void Clear() { environments_.clear(); }

 private:
  // Called by GetEnv() when it calls lazily creates a new environment, to
  // globally track stats about how many such environments are created by
  // CompilationEnvironments.
  static void DefaultEnvCreatedByCompilationEnvironments(
      std::string_view env_type);

  // Called by AddEnv(), to globally track stats about how many environments
  // are added to CompilationEnvironments.
  static void EnvAdded(std::string_view env_type);

  // Implements the part of AddEnv() after the ProcessNewEnv() call.
  void AddProcessedEnv(std::unique_ptr<tsl::protobuf::Message> env);

  absl::flat_hash_map<const tsl::protobuf::Descriptor*,
                      std::unique_ptr<tsl::protobuf::Message>>
      environments_;
};

// ----- Template implementation below -----

// Make sure no one tries to specialize ProcessNewEnv() for raw
// tsl::protobuf::Message. Specialization should always be for a specific
// type of proto message.
template <>
std::unique_ptr<tsl::protobuf::Message> CompilationEnvironments::ProcessNewEnv(
    std::unique_ptr<tsl::protobuf::Message> env) = delete;

template <typename T>
void CompilationEnvironments::AddEnv(std::unique_ptr<T> env) {
  AddProcessedEnv(ProcessNewEnv<T>(std::move(env)));
}

template <typename T>
T& CompilationEnvironments::GetMutableEnv() {
  auto descriptor = T::descriptor();
  auto it = environments_.find(descriptor);
  if (it == environments_.end()) {
    AddEnv<T>(nullptr);
    DefaultEnvCreatedByCompilationEnvironments(descriptor->full_name());
    it = environments_.find(descriptor);
  }
  return tensorflow::down_cast<T&>(*it->second);
}

template <typename T>
const T& CompilationEnvironments::GetEnv() {
  return GetMutableEnv<T>();
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILATION_ENVIRONMENTS_H_
