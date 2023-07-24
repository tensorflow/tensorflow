/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_

#include <map>

#include "absl/base/macros.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/fft.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/stream_executor/plugin.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace stream_executor {

namespace internal {
class StreamExecutorInterface;
}

// The PluginRegistry is a singleton that maintains the set of registered
// "support library" plugins. Currently, there are four kinds of plugins:
// BLAS, DNN, and FFT. Each interface is defined in the corresponding
// gpu_{kind}.h header.
//
// At runtime, a StreamExecutor object will query the singleton registry to
// retrieve the plugin kind that StreamExecutor was configured with (refer to
// the StreamExecutor and PluginConfig declarations).
//
// Plugin libraries are best registered using REGISTER_MODULE_INITIALIZER,
// but can be registered at any time. When registering a DSO-backed plugin, it
// is usually a good idea to load the DSO at registration time, to prevent
// late-loading from distorting performance/benchmarks as much as possible.
class PluginRegistry {
 public:
  typedef blas::BlasSupport* (*BlasFactory)(internal::StreamExecutorInterface*);
  typedef dnn::DnnSupport* (*DnnFactory)(internal::StreamExecutorInterface*);
  typedef fft::FftSupport* (*FftFactory)(internal::StreamExecutorInterface*);

  // Gets (and creates, if necessary) the singleton PluginRegistry instance.
  static PluginRegistry* Instance();

  // Registers the specified factory with the specified platform.
  // Returns a non-successful status if the factory has already been registered
  // with that platform (but execution should be otherwise unaffected).
  template <typename FactoryT>
  tsl::Status RegisterFactory(Platform::Id platform_id, PluginId plugin_id,
                              const std::string& name, FactoryT factory);

  // Registers the specified factory as usable by _all_ platform types.
  // Reports errors just as RegisterFactory.
  template <typename FactoryT>
  tsl::Status RegisterFactoryForAllPlatforms(PluginId plugin_id,
                                             const std::string& name,
                                             FactoryT factory);

  // TODO(b/22689637): Setter for temporary mapping until all users are using
  // MultiPlatformManager / PlatformId.
  void MapPlatformKindToId(PlatformKind platform_kind,
                           Platform::Id platform_id);

  // Potentially sets the plugin identified by plugin_id to be the default
  // for the specified platform and plugin kind. If this routine is called
  // multiple types for the same PluginKind, the PluginId given in the last call
  // will be used.
  bool SetDefaultFactory(Platform::Id platform_id, PluginKind plugin_kind,
                         PluginId plugin_id);

  // Return true if the factory/id has been registered for the
  // specified platform and plugin kind and false otherwise.
  bool HasFactory(Platform::Id platform_id, PluginKind plugin_kind,
                  PluginId plugin) const;

  // Retrieves the factory registered for the specified kind,
  // or a tsl::Status on error.
  template <typename FactoryT>
  tsl::StatusOr<FactoryT> GetFactory(Platform::Id platform_id,
                                     PluginId plugin_id);

  // TODO(b/22689637): Deprecated/temporary. Will be deleted once all users are
  // on MultiPlatformManager / PlatformId.
  template <typename FactoryT>
  ABSL_DEPRECATED("Use MultiPlatformManager / PlatformId instead.")
  tsl::StatusOr<FactoryT> GetFactory(PlatformKind platform_kind,
                                     PluginId plugin_id);

 private:
  // Containers for the sets of registered factories, by plugin kind.
  struct PluginFactories {
    std::map<PluginId, BlasFactory> blas;
    std::map<PluginId, DnnFactory> dnn;
    std::map<PluginId, FftFactory> fft;
  };

  // Simple structure to hold the currently configured default plugins (for a
  // particular Platform).
  struct DefaultFactories {
    DefaultFactories();
    PluginId blas, dnn, fft;
  };

  PluginRegistry();

  // Actually performs the work of registration.
  template <typename FactoryT>
  tsl::Status RegisterFactoryInternal(PluginId plugin_id,
                                      const std::string& plugin_name,
                                      FactoryT factory,
                                      std::map<PluginId, FactoryT>* factories);

  // Actually performs the work of factory retrieval.
  template <typename FactoryT>
  tsl::StatusOr<FactoryT> GetFactoryInternal(
      PluginId plugin_id, const std::map<PluginId, FactoryT>& factories,
      const std::map<PluginId, FactoryT>& generic_factories) const;

  // Returns true if the specified plugin has been registered with the specified
  // platform factories. Unlike the other overload of this method, this does
  // not implicitly examine the default factory lists.
  bool HasFactory(const PluginFactories& factories, PluginKind plugin_kind,
                  PluginId plugin) const;

  // The singleton itself.
  static PluginRegistry* instance_;

  // TODO(b/22689637): Temporary mapping until all users are using
  // MultiPlatformManager / PlatformId.
  std::map<PlatformKind, Platform::Id> platform_id_by_kind_;

  // The set of registered factories, keyed by platform ID.
  std::map<Platform::Id, PluginFactories> factories_;

  // Plugins supported for all platform kinds.
  PluginFactories generic_factories_;

  // The sets of default factories, keyed by platform ID.
  std::map<Platform::Id, DefaultFactories> default_factories_;

  // Lookup table for plugin names.
  std::map<PluginId, std::string> plugin_names_;

  SE_DISALLOW_COPY_AND_ASSIGN(PluginRegistry);
};

// Explicit specializations are defined in plugin_registry.cc.
#define DECLARE_PLUGIN_SPECIALIZATIONS(FACTORY_TYPE)                         \
  template <>                                                                \
  tsl::Status PluginRegistry::RegisterFactory<PluginRegistry::FACTORY_TYPE>( \
      Platform::Id platform_id, PluginId plugin_id, const std::string& name, \
      PluginRegistry::FACTORY_TYPE factory);                                 \
  template <>                                                                \
  tsl::StatusOr<PluginRegistry::FACTORY_TYPE> PluginRegistry::GetFactory(    \
      Platform::Id platform_id, PluginId plugin_id);                         \
  template <>                                                                \
  tsl::StatusOr<PluginRegistry::FACTORY_TYPE> PluginRegistry::GetFactory(    \
      PlatformKind platform_kind, PluginId plugin_id)

DECLARE_PLUGIN_SPECIALIZATIONS(BlasFactory);
DECLARE_PLUGIN_SPECIALIZATIONS(DnnFactory);
DECLARE_PLUGIN_SPECIALIZATIONS(FftFactory);
#undef DECL_PLUGIN_SPECIALIZATIONS

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_
