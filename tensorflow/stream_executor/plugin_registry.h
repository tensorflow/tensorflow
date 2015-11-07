#ifndef TENSORFLOW_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_

#include <map>

#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/fft.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/plugin.h"
#include "tensorflow/stream_executor/rng.h"

namespace perftools {
namespace gputools {

namespace internal {
class StreamExecutorInterface;
}

// The PluginRegistry is a singleton that maintains the set of registered
// "support library" plugins. Currently, there are four kinds of plugins:
// BLAS, DNN, FFT, and RNG. Each interface is defined in the corresponding
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
  typedef rng::RngSupport* (*RngFactory)(internal::StreamExecutorInterface*);

  // Gets (and creates, if necessary) the singleton PluginRegistry instance.
  static PluginRegistry* Instance();

  // Registers the specified factory with the specified platform.
  // Returns a non-successful status if the factory has already been registered
  // with that platform (but execution should be otherwise unaffected).
  template <typename FactoryT>
  port::Status RegisterFactory(Platform::Id platform_id, PluginId plugin_id,
                               const string& name, FactoryT factory);

  // Registers the specified factory as usable by _all_ platform types.
  // Reports errors just as RegisterFactory.
  template <typename FactoryT>
  port::Status RegisterFactoryForAllPlatforms(PluginId plugin_id,
                                              const string& name,
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
  // or a port::Status on error.
  template <typename FactoryT>
  port::StatusOr<FactoryT> GetFactory(Platform::Id platform_id,
                                      PluginId plugin_id);

  // TODO(b/22689637): Deprecated/temporary. Will be deleted once all users are
  // on MultiPlatformManager / PlatformId.
  template <typename FactoryT>
  port::StatusOr<FactoryT> GetFactory(PlatformKind platform_kind,
                                      PluginId plugin_id);

 private:
  // Containers for the sets of registered factories, by plugin kind.
  struct PluginFactories {
    std::map<PluginId, BlasFactory> blas;
    std::map<PluginId, DnnFactory> dnn;
    std::map<PluginId, FftFactory> fft;
    std::map<PluginId, RngFactory> rng;
  };

  // Simple structure to hold the currently configured default plugins (for a
  // particular Platform).
  struct DefaultFactories {
    DefaultFactories();
    PluginId blas, dnn, fft, rng;
  };

  PluginRegistry();

  // Actually performs the work of registration.
  template <typename FactoryT>
  port::Status RegisterFactoryInternal(PluginId plugin_id,
                                       const string& plugin_name,
                                       FactoryT factory,
                                       std::map<PluginId, FactoryT>* factories);

  // Actually performs the work of factory retrieval.
  template <typename FactoryT>
  port::StatusOr<FactoryT> GetFactoryInternal(
      PluginId plugin_id, const std::map<PluginId, FactoryT>& factories,
      const std::map<PluginId, FactoryT>& generic_factories) const;

  // Returns true if the specified plugin has been registered with the specified
  // platform factories. Unlike the other overload of this method, this does
  // not implicitly examine the default factory lists.
  bool HasFactory(const PluginFactories& factories, PluginKind plugin_kind,
                  PluginId plugin) const;

  // As this object is a singleton, a global mutex can be used for static and
  // instance protection.
  static mutex mu_;

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
  std::map<PluginId, string> plugin_names_;

  SE_DISALLOW_COPY_AND_ASSIGN(PluginRegistry);
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLUGIN_REGISTRY_H_
