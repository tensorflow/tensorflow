/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/stream_executor/plugin.h"

namespace perftools {
namespace gputools {

// Mostly-arbitrary ID only used as a sentinel "not otherwise initialized"
// value. This value should never [need to] be specified aside by initialization
// functions defined in this file and in PluginRegistry.
PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(PluginConfig::kDefault);

PluginConfig::PluginConfig()
    : blas_(kDefault), dnn_(kDefault), fft_(kDefault), rng_(kDefault) {}

bool PluginConfig::operator==(const PluginConfig& rhs) const {
  return blas_ == rhs.blas_ && dnn_ == rhs.dnn_ && fft_ == rhs.fft_ &&
         rng_ == rhs.rng_;
}

PluginConfig& PluginConfig::SetBlas(PluginId blas) {
  blas_ = blas;
  return *this;
}

PluginConfig& PluginConfig::SetDnn(PluginId dnn) {
  dnn_ = dnn;
  return *this;
}

PluginConfig& PluginConfig::SetFft(PluginId fft) {
  fft_ = fft;
  return *this;
}

PluginConfig& PluginConfig::SetRng(PluginId rng) {
  rng_ = rng;
  return *this;
}

}  // namespace gputools
}  // namespace perftools
