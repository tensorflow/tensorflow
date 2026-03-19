/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_

#include <string>
#include <utility>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {

class StreamExecutor;

// Enumeration to list the supported types of plugins / support libraries.
enum class PluginKind {
  kBlas,
  kDnn,
  kFft,
};

// The PluginRegistry is a singleton that maintains the set of registered
// "support library" plugins. Currently, there are four kinds of plugins:
// BLAS, DNN, and FFT. Each interface is defined in the corresponding
// gpu_{kind}.h header.
//
// At runtime, a StreamExecutor object will query the singleton registry to
// retrieve the plugin kind that StreamExecutor was configured with (refer to
// the StreamExecutor declarations).
//
// Plugin libraries are best registered using REGISTER_MODULE_INITIALIZER,
// but can be registered at any time. When registering a DSO-backed plugin, it
// is usually a good idea to load the DSO at registration time, to prevent
// late-loading from distorting performance/benchmarks as much as possible.
class PluginRegistry {
 public:
  typedef blas::BlasSupport* (*BlasFactory)(StreamExecutor*);
  typedef dnn::DnnSupport* (*DnnFactory)(StreamExecutor*);
  typedef fft::FftSupport* (*FftFactory)(StreamExecutor*);

  // Gets (and creates, if necessary) the singleton PluginRegistry instance.
  static PluginRegistry* Instance();

  // Registers the specified factory with the specified platform.
  // Returns a non-successful status if the factory has already been registered
  // with that platform (but execution should be otherwise unaffected).
  template <typename FactoryT>
  absl::Status RegisterFactory(Platform::Id platform_id,
                               const std::string& name, FactoryT factory);

  // Return true if the factory/kind has been registered for the
  // specified platform and plugin kind and false otherwise.
  bool HasFactory(Platform::Id platform_id, PluginKind plugin_kind) const;

  // Retrieves the factory registered for the specified kind,
  // or a absl::Status on error.
  template <typename FactoryT>
  absl::StatusOr<FactoryT> GetFactory(Platform::Id platform_id) const;

 private:
  using AnyFactory = std::variant<BlasFactory, DnnFactory, FftFactory>;

  PluginRegistry() = default;
  PluginRegistry(const PluginRegistry&) = delete;
  void operator=(const PluginRegistry&) = delete;

  mutable absl::Mutex registry_mutex_;
  // The set of registered factories, keyed by platform ID and plugin kind.
  absl::flat_hash_map<std::pair<Platform::Id, PluginKind>, AnyFactory>
      factories_ ABSL_GUARDED_BY(registry_mutex_);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_
