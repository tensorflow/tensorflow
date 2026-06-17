/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_ONEDNN_LAYER_NORM_H_
#define XLA_SERVICE_CPU_ONEDNN_LAYER_NORM_H_

#include "dnnl.hpp"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"

namespace xla {
namespace cpu {

void ExecuteOneDnnLayerNorm(OneDnnNormConfig ln_config,
                            const dnnl::engine& cpu_engine,
                            dnnl::stream& onednn_stream,
                            OneDnnPrimResources& resources);

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_ONEDNN_LAYER_NORM_H_
