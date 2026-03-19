/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_ONEDNN_CONVOLUTION_H_
#define XLA_SERVICE_CPU_ONEDNN_CONVOLUTION_H_

#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_util.h"

namespace xla {
namespace cpu {

constexpr auto kOnednnConvConfig = BackendConfigOneofCase::kOnednnConvConfig;

void ExecuteOneDnnConvolution(absl::Span<MemrefInfoHandler> arguments,
                              absl::Span<MemrefInfoHandler> results,
                              OneDnnConvolutionConfig conv_config,
                              const dnnl::engine& cpu_engine,
                              dnnl::stream& onednn_stream,
                              OneDnnResources& resources);

template <>
struct PrimitiveTrait<kOnednnConvConfig> {
  using pointer_type = xla::cpu::OneDnnConvolutionConfig*;
  using primitive_desc = dnnl::convolution_forward::primitive_desc;
  static const BackendConfigOneofCase kConfigVal = kOnednnConvConfig;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_ONEDNN_CONVOLUTION_H_
