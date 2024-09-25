/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SYMBOL_REPOSITORY_H_
#define XLA_SERVICE_SYMBOL_REPOSITORY_H_

// Functionality to do lookups in HLO repositories. See export_hlo.h for
// uploads.

#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/xla.pb.h"

namespace xla {

// Different backends that repositories might store symbols for. This enum could
// change to a string in the future if required, but ideally repositories only
// care about the class of hardware, not the specific make/model and so an enum
// is fine.
enum class BackendType {
  kCpu,
  kGpu,
  kTpu,
};

// Dummy struct for individual backends to add their data to.
struct BackendSpecificData {
  virtual ~BackendSpecificData() = default;
};

// A module and some collected metadata that allow for pure compilation of an
// HLO module. Implementations may want to subclass to add additional
// functionality or data.
struct HloModuleAndMetadata {
  virtual ~HloModuleAndMetadata() = default;

  std::unique_ptr<HloModule> hlo_module;
  std::unique_ptr<Compiler::TargetConfig> target_config;
  // Use static_cast to cast this to a concrete type.
  std::unique_ptr<BackendSpecificData> backend_specific_data;
};

// Looks up HLO in a repository. The only non-dummy implementation is
// Google-internal as of 2023-10.
class SymbolRepository {
 public:
  virtual ~SymbolRepository() = default;
  virtual absl::StatusOr<std::unique_ptr<HloModuleAndMetadata>> Lookup(
      absl::string_view symbol_reference, BackendType backend) const = 0;
};

// Registry for SymbolRepository implementations.
class SymbolRepositoryRegistry {
 public:
  void Register(const std::string& name,
                std::unique_ptr<SymbolRepository> repo) {
    absl::MutexLock lock(&mu_);
    VLOG(1) << "Registering SymbolRepository " << name;
    repo_[name] = std::move(repo);
  }

  SymbolRepository* repo(absl::string_view name) {
    absl::MutexLock lock(&mu_);
    const auto it = repo_.find(name);
    if (it == repo_.end()) {
      return nullptr;
    }

    return it->second.get();
  }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<SymbolRepository>> repo_
      ABSL_GUARDED_BY(mu_);
};

inline SymbolRepositoryRegistry& GetGlobalSymbolRepositoryRegistry() {
  static auto* const registry = new SymbolRepositoryRegistry;
  return *registry;
}

// Entry points start here.

inline absl::StatusOr<std::unique_ptr<HloModuleAndMetadata>>
LookupSymbolInRepository(absl::string_view repository,
                         absl::string_view symbol_reference,
                         BackendType backend) {
  if (SymbolRepository* repo =
          GetGlobalSymbolRepositoryRegistry().repo(repository);
      repo != nullptr) {
    return repo->Lookup(symbol_reference, backend);
  }

  return nullptr;
}

}  // namespace xla

#endif  // XLA_SERVICE_SYMBOL_REPOSITORY_H_
