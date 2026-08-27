/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdlib>
#include <cstring>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.h"

namespace tensorflow {
namespace metal {
namespace {

// Set TF_DISABLE_METAL=1 to keep the backend out of the process entirely.
//
// Worth having for a device backend that is new in this release: it lets a
// user fall back to CPU without reinstalling, and lets a bug report separate
// "Metal is involved" from "Metal is the cause" in one run.
bool MetalDisabledByEnvironment() {
  const char* value = std::getenv("TF_DISABLE_METAL");
  if (value == nullptr) return false;
  return std::strcmp(value, "0") != 0 && value[0] != '\0';
}

absl::Status RegisterMetalPlugin() {
  if (MetalDisabledByEnvironment()) {
    LOG(INFO) << "Metal: backend disabled by TF_DISABLE_METAL.";
    return absl::OkStatus();
  }

  // The in-process form of plugin registration. The dlopen path resolves
  // SE_InitPlugin and friends out of a shared object; here the backend is
  // linked in, so the same struct is filled with a direct function pointer and
  // core does the identical work: register a PluggableDeviceFactory for the
  // device type and wire up device-to-device tensor copies.
  //
  // Only the device module is supplied. Kernels are registered separately
  // through the Kernel C API, and the graph optimizer and profiler modules are
  // not implemented yet, which core treats as absent rather than as an error.
  PluggableDeviceInit_Api api;
  api.init_plugin_fn = MetalInitPlugin;
  return RegisterPluggableDevicePlugin(&api);
}

// Registered from a static initializer, the same mechanism
// REGISTER_LOCAL_DEVICE_FACTORY uses for the built-in CPU and CUDA devices,
// so the Metal device exists as soon as the framework is loaded rather than
// only after some later explicit call. MetalInitPlugin itself only fills
// function tables; no Metal API is touched until core asks for the device
// count, so this does not pull the Metal framework into process startup.
//
// The target is alwayslink so the linker keeps this translation unit even
// though nothing references it.
const bool kMetalPluginRegistered = [] {
  const absl::Status status = RegisterMetalPlugin();
  if (!status.ok()) {
    // Deliberately not fatal. A machine with no usable Metal device, or a
    // registration clash with another GPU plugin, should degrade to CPU rather
    // than make `import tensorflow` fail.
    LOG(ERROR) << "Metal: could not register the Metal PluggableDevice: "
               << status.message() << ". Falling back to CPU.";
    return false;
  }
  return true;
}();

}  // namespace
}  // namespace metal
}  // namespace tensorflow
