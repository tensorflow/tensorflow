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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_

#include "tensorflow/core/common_runtime/next_pluggable_device/plugin_coordination_service_agent.h"

#ifndef TF_NEXT_PLUGGABLE_DEVICE_USE_C_API
#include "tensorflow/core/common_runtime/next_pluggable_device/direct_plugin_coordination_service_agent.h"
#else
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c_plugin_coordination_service_agent.h"
#endif  // TF_NEXT_PLUGGABLE_DEVICE_USE_C_API

namespace tensorflow {

inline PluginCoordinationServiceAgent* CreatePluginCoordinationServiceAgent(
    void* agent) {
#ifndef TF_NEXT_PLUGGABLE_DEVICE_USE_C_API
  return new DirectPluginCoordinationServiceAgent(agent);
#else
  return new CPluginCoordinationServiceAgent(agent);
#endif  // TF_NEXT_PLUGGABLE_DEVICE_USE_C_API
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_HELPER_H_
