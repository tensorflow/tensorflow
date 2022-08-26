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

#include "tensorflow/compiler/xla/service/compilation_environments.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace {

// A global singleton stats object for implementing CompilationEnvironments::{
// DefaultEnvCreated(), DefaultEnvCreatedByCompilationEnvironments(),
// EnvAdded()}.
class GlobalCompEnvStats {
 public:
  static GlobalCompEnvStats& GetSingleton() {
    static GlobalCompEnvStats* singleton = new GlobalCompEnvStats();

    return *singleton;
  }

  void DefaultEnvCreated(std::string_view env_type) ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock l(&mu_);
      ++stats_[std::string(env_type)].default_env_created;
    }
    VLOG(1) << "New GlobalCompEnvStats value: " << ToString();
  }

  void DefaultEnvCreatedByCompilationEnvironments(std::string_view env_type)
      ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock l(&mu_);
      ++stats_[std::string(env_type)]
            .default_env_created_by_compilation_environments;
    }
    VLOG(1) << "New GlobalCompEnvStats value: " << ToString();
  }

  void EnvAdded(std::string_view env_type) ABSL_LOCKS_EXCLUDED(mu_) {
    {
      absl::MutexLock l(&mu_);
      ++stats_[std::string(env_type)].env_added;
    }
    VLOG(1) << "New GlobalCompEnvStats value: " << ToString();
  }

  std::string ToString() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock l(&mu_);
    return absl::StrJoin(
        stats_, "; ",
        [](std::string* out, const StatMap::value_type& env_stats_pair) {
          absl::StrAppend(out, env_stats_pair.first, ": { ",
                          env_stats_pair.second.ToString(), " }");
        });
  }

 private:
  struct PerEnvStats {
    std::string ToString() const {
      return absl::StrCat(
          "# default envs created: ", default_env_created, " ",
          "# default envs created by CompilationEnvironments: ",
          default_env_created_by_compilation_environments, " ",
          "# envs added to CompilationEnvironments: ", env_added);
    }

    unsigned default_env_created = 0;
    unsigned default_env_created_by_compilation_environments = 0;
    unsigned env_added = 0;
  };

  using StatMap = absl::flat_hash_map<std::string, PerEnvStats>;

  GlobalCompEnvStats() = default;
  GlobalCompEnvStats(const GlobalCompEnvStats&) = delete;
  GlobalCompEnvStats& operator=(const GlobalCompEnvStats&) = delete;
  GlobalCompEnvStats(GlobalCompEnvStats&&) = delete;
  GlobalCompEnvStats& operator=(GlobalCompEnvStats&&) = delete;

  mutable absl::Mutex mu_;
  StatMap stats_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

void CompilationEnvironments::DefaultEnvCreated(std::string_view env_type) {
  GlobalCompEnvStats::GetSingleton().DefaultEnvCreated(env_type);
}

void CompilationEnvironments::DefaultEnvCreatedByCompilationEnvironments(
    std::string_view env_type) {
  GlobalCompEnvStats::GetSingleton().DefaultEnvCreatedByCompilationEnvironments(
      env_type);
}

void CompilationEnvironments::EnvAdded(std::string_view env_type) {
  GlobalCompEnvStats::GetSingleton().EnvAdded(env_type);
}

CompilationEnvironments& CompilationEnvironments::operator=(
    const CompilationEnvironments& rhs) {
  Clear();
  for (const auto& descriptor_message_pair : rhs.environments_) {
    auto env = absl::WrapUnique(descriptor_message_pair.second->New());
    env->CopyFrom(*descriptor_message_pair.second);
    environments_.insert({descriptor_message_pair.first, std::move(env)});
  }
  return *this;
}

void CompilationEnvironments::AddEnv(
    std::unique_ptr<tensorflow::protobuf::Message> env) {
  auto descriptor = env->GetDescriptor();
  if (environments_.contains(descriptor)) {
    LOG(WARNING) << "Replacing CompilationEnvironment of type "
                 << descriptor->full_name();
  }

  environments_.insert({descriptor, std::move(env)});
  EnvAdded(descriptor->full_name());
}

}  // namespace xla
