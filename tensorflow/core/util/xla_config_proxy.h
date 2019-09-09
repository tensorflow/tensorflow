/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_XLA_CONFIG_PROXY_H
#define TENSORFLOW_CORE_UTIL_XLA_CONFIG_PROXY_H

#include <functional>
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A proxy class of XLA config.
class XlaConfigProxy {
 public:
  // This class provides a means for updating configs (mainly ConfigProto)
  // according to runtime environment variable flags. As the flags may be
  // maintained outside the Tensorflow core and they may or may not exist
  // depending on the build configuration, here implements a registration
  // mechanism to register their config setters for query.
  template <typename ConfigType>
  class ConfigSetterRegistry {
   public:
    static ConfigSetterRegistry* Global() {
      static ConfigSetterRegistry* config_registry = new ConfigSetterRegistry;
      return config_registry;
    }

    // Register a setter for ConfigType.
    void Register(std::function<bool(ConfigType&)> setter) {
      CHECK(!a_setter_);
      a_setter_ = std::move(setter);
    }

    // Invoke the registered setter to update the ConfigType value. Return true
    // if the value is updated.
    bool Update(ConfigType& value) {
      if (!a_setter_) {
        return true;
      }
      return a_setter_(value);
    }

   private:
    ConfigSetterRegistry() = default;
    ConfigSetterRegistry(const ConfigSetterRegistry&) = delete;
    ConfigSetterRegistry& operator=(const ConfigSetterRegistry&) = delete;

   private:
    std::function<bool(ConfigType&)> a_setter_;
  };

  template <typename ConfigType>
  class ConfigSetterRegistration {
   public:
    ConfigSetterRegistration(std::function<bool(ConfigType&)> setter) {
      ConfigSetterRegistry<ConfigType>::Global()->Register(std::move(setter));
    }
  };

 public:
  static OptimizerOptions::GlobalJitLevel GetGlobalJitLevel(
      OptimizerOptions::GlobalJitLevel jit_level_in_session_opts);
};

#define REGISTER_XLA_CONFIG_SETTER(ConfigType, setter)                      \
  static ::tensorflow::XlaConfigProxy::ConfigSetterRegistration<ConfigType> \
      unique_xla_config_setter_ctr_##__COUNTER__(setter)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_XLA_CONFIG_PROXY_H
