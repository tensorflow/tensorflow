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
