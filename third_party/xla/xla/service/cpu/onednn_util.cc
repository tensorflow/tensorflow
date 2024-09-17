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
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/onednn_util.h"

#define EIGEN_USE_THREADS

namespace xla {
namespace cpu {

std::unique_ptr<tsl::OneDnnThreadPool> CreateOneDnnThreadPool(
    const Eigen::ThreadPoolDevice* threadpool_device) {
#ifndef ENABLE_ONEDNN_OPENMP
  if (threadpool_device != nullptr) {
    return std::make_unique<tsl::OneDnnThreadPool>(threadpool_device->getPool(),
                                                   false);
  }
#endif  // !ENABLE_ONEDNN_OPENMP
  return nullptr;
}

dnnl::stream MakeOneDnnStream(
    const dnnl::engine& cpu_engine,
    dnnl::threadpool_interop::threadpool_iface* thread_pool) {
  return (thread_pool != nullptr)
             ? dnnl::threadpool_interop::make_stream(cpu_engine, thread_pool)
             : dnnl::stream(cpu_engine);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
