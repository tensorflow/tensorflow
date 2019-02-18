/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_GRAPH_CACHING_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_OPS_GRAPH_CACHING_UTIL_H_

#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {
using PoplarTensorSignature = std::pair<poplar::Type, std::vector<std::size_t>>;
namespace graph_caching_util {
PoplarTensorSignature GetPoplarTensorSignature(const poplar::Tensor& tensor);

}  // namespace graph_caching_util
}  // namespace poplarplugin
}  // namespace xla

#endif