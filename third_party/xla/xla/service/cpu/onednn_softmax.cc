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

#include "xla/service/cpu/onednn_softmax.h"

#include <string>
#include <unordered_map>

#include "absl/base/attributes.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "xla/executable_run_options.h"
#include "xla/service/cpu/backend_config.pb.h"
// Below must come after `onednn_threadpool.h`
#include "unsupported/Eigen/CXX11/Tensor"  // NOLINT

namespace xla {
namespace cpu {

void ExecuteOneDnnSoftmax(absl::Span<MemrefInfoHandler> arguments,
                          absl::Span<MemrefInfoHandler> results,
                          OneDnnSoftmaxConfig softmax_config,
                          const dnnl::engine& cpu_engine,
                          dnnl::stream& onednn_stream,
                          OneDnnResources& resources) {
  MemrefInfo input_minfo(arguments[0].get());
  MemrefInfo result_minfo(results[0].get());

  auto src_md = input_minfo.GetOneDnnMemDesc();
  auto dst_md = result_minfo.GetOneDnnMemDesc();

  resources.src_mem = dnnl::memory(src_md, cpu_engine, input_minfo.Data());
  resources.dst_mem = dnnl::memory(dst_md, cpu_engine, result_minfo.Data());

  int axis = softmax_config.softmax_axis();

  auto softmax_pd = dnnl::softmax_forward::primitive_desc(
      cpu_engine, dnnl::prop_kind::forward_inference,
      dnnl::algorithm::softmax_accurate, src_md, dst_md, axis);

  resources.primitive = dnnl::primitive(softmax_pd);

  std::unordered_map<int, dnnl::memory> softmax_args = {
      {DNNL_ARG_SRC, resources.src_mem},
      {DNNL_ARG_DST, resources.dst_mem},
  };

  resources.primitive.execute(onednn_stream, softmax_args);
}

}  // namespace cpu
}  // namespace xla
