/* Copyright 2019 TensorFlow Ltd

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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HASH_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_HASH_H_
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {
namespace poplarplugin {
namespace hash_util {

inline std::size_t hash() { return 0; }

template <typename T>
std::size_t hash(const T& v) {
  return std::hash<T>()(v);
}

template <typename T, typename... Args>
std::size_t hash(const T& first, const Args&... args) {
  return tensorflow::Hash64Combine(hash(first), hash(args...));
}

}  // namespace hash_util
}  // namespace poplarplugin
}  // namespace xla

#endif
