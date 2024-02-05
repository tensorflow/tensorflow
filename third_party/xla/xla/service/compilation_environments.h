/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COMPILATION_ENVIRONMENTS_H_
#define XLA_SERVICE_COMPILATION_ENVIRONMENTS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <typeindex>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xla/statusor.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/protobuf.h"

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
  using ProcessNewEnvFn =
      std::function<StatusOr<std::unique_ptr<tsl::protobuf::Message>>(
          std::unique_ptr<tsl::protobuf::Message>)>;

  CompilationEnvironments() = default;
  CompilationEnvironments(const CompilationEnvironments& rhs) { *this = rhs; }
  CompilationEnvironments& operator=(const CompilationEnvironments& rhs);
  ~CompilationEnvironments() = default;

  // Deserializes the given CompilationEnvironments proto.
  static StatusOr<std::unique_ptr<CompilationEnvironments>> CreateFromProto(
      const CompilationEnvironmentsProto& proto);

  // Whenever an environment is added to CompilationEnvironments, even when
  // GetEnv() adds a lazily initialized one, it is passed to the function
  // registered by this method, corresponding to the environment's proto
  // descriptor. The result is the environment that is used by
  // CompilationEnvironments. This allows environment authors to
  // do things like populate missing fields in an added environment.
  //
  // Users of CompilationEnvironments must register their `ProcessNewEnvFn`
  // function via this method for each type of CompilationEnvironment they wish
  // to use in code.
  //
  // The input env to a ProcessNewEnvFn may be null.
  //
  // REQUIRES:
  // - The output is *not* allowed to be null, even for null input.
  static void RegisterProcessNewEnvFn(
      const tsl::protobuf::Descriptor* descriptor,
      ProcessNewEnvFn process_new_env);

  // Adds env to the list of CompilationEnvironments. If an environment with
  // the same proto descriptor has already been added, env will replace it.
  //
  // All added environments are processed via registered ProcessNewEnvFns. If
  // such a function was not regitered for env's proto descriptor or env's
  // proto type is unknown, an error will be returned.
  Status AddEnv(std::unique_ptr<tsl::protobuf::Message> env);

  // Returns the CompilationEnvironment corresponding to T. If such an
  // environment has not been added, ProcessNewEnvFn(nullptr) will be added and
  // returned.
  //
  // GetMutableEnv()/GetEnv() are not const because they can perform lazy
  // initialization, thereby modifying the CompilationEnvironments's data
  // members.
  template <typename T>
  T& GetMutableEnv();
  template <typename T>
  const T& GetEnv();
  template <typename T>
  bool HasEnv();

  // Removes all added environments.
  void Clear() { environments_.clear(); }

  // Serializes this CompilationEnvironments into a protobuf message.
  CompilationEnvironmentsProto ToProto() const;

 private:
  // Returns the ProcessNewEnvFn for the given env type. Returns nullptr if no
  // ProcessNewEnvFn has been registered for the env type.
  static ProcessNewEnvFn GetProcessNewEnvFn(
      const tsl::protobuf::Descriptor& descriptor);

  // Called by GetEnv(), when it lazily creates a new environment, to globally
  // track stats about how many such environments are created by
  // CompilationEnvironments.
  static void DefaultEnvCreatedByCompilationEnvironments(
      std::string_view env_type);

  // Called by AddEnv(), to globally track stats about how many environments
  // are added to CompilationEnvironments.
  static void EnvAdded(std::string_view env_type);

  Status AddEnvImpl(const tsl::protobuf::Descriptor& descriptor,
                    std::unique_ptr<tsl::protobuf::Message> env);

  absl::flat_hash_map<const tsl::protobuf::Descriptor*,
                      std::unique_ptr<tsl::protobuf::Message>>
      environments_;
};

// ----- Template implementation below -----

template <typename T>
T& CompilationEnvironments::GetMutableEnv() {
  auto descriptor = T::descriptor();
  auto it = environments_.find(descriptor);
  if (it == environments_.end()) {
    TF_CHECK_OK(AddEnvImpl(*descriptor, nullptr));
    DefaultEnvCreatedByCompilationEnvironments(descriptor->full_name());
    it = environments_.find(descriptor);
  }

  return tensorflow::down_cast<T&>(*it->second);
}

template <typename T>
const T& CompilationEnvironments::GetEnv() {
  return GetMutableEnv<T>();
}

template <typename T>
bool CompilationEnvironments::HasEnv() {
  auto descriptor = T::descriptor();
  return environments_.find(descriptor) != environments_.end();
}

}  // namespace xla

#endif  // XLA_SERVICE_COMPILATION_ENVIRONMENTS_H_
