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

#ifndef TENSORFLOW_STREAM_EXECUTOR_PLUGIN_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLUGIN_H_

namespace stream_executor {

// A plugin ID is a unique identifier for each registered plugin type.
typedef void* PluginId;

// Helper macro to define a plugin ID. To be used only inside plugin
// implementation files. Works by "reserving" an address/value (guaranteed to be
// unique) inside a process space.
#define PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(ID_VAR_NAME) \
  namespace {                                         \
  int plugin_id_value;                                \
  }                                                   \
  const PluginId ID_VAR_NAME = &plugin_id_value;

// kNullPlugin denotes an invalid plugin identifier.
extern const PluginId kNullPlugin;

// Enumeration to list the supported types of plugins / support libraries.
enum class PluginKind {
  kInvalid,
  kBlas,
  kDnn,
  kFft,
  kRng,
};

// A PluginConfig describes the set of plugins to be used by a StreamExecutor
// instance. Each plugin is defined by an arbitrary identifier, usually best set
// to the address static member in the implementation (to avoid conflicts).
//
// A PluginConfig may be passed to the StreamExecutor constructor - the plugins
// described therein will be used to provide BLAS, DNN, FFT, and RNG
// functionality. Platform-appropriate defaults will be used for any un-set
// libraries. If a platform does not support a specified plugin (ex. cuBLAS on
// an OpenCL executor), then an error will be logged and no plugin operations
// will succeed.
//
// The StreamExecutor BUILD target does not link ANY plugin libraries - even
// common host fallbacks! Any plugins must be explicitly linked by dependent
// targets. See the cuda, opencl and host BUILD files for implemented plugin
// support (search for "plugin").
class PluginConfig {
 public:
  // Value specifying the platform's default option for that plugin.
  static const PluginId kDefault;

  // Initializes all members to the default options.
  PluginConfig();

  bool operator==(const PluginConfig& rhs) const;

  // Sets the appropriate library kind to that passed in.
  PluginConfig& SetBlas(PluginId blas);
  PluginConfig& SetDnn(PluginId dnn);
  PluginConfig& SetFft(PluginId fft);
  PluginConfig& SetRng(PluginId rng);

  PluginId blas() const { return blas_; }
  PluginId dnn() const { return dnn_; }
  PluginId fft() const { return fft_; }
  PluginId rng() const { return rng_; }

 private:
  PluginId blas_, dnn_, fft_, rng_;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLUGIN_H_
