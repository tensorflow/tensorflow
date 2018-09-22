/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_options.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace xla {
namespace gpu {

bool ConvUseLayoutHeuristic(const HloModuleConfig& config) {
  return !config.debug_options().xla_backend_extra_options().count(
      "xla_gpu_experimental_conv_disable_layout_heuristic");
}

}  // namespace gpu
}  // namespace xla
