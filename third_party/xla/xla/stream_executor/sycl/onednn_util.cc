/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/onednn_util.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/str_util.h"

namespace stream_executor {
namespace sycl {

static absl::Mutex engine_map_mutex(absl::kConstInit);

dnnl::memory::dims ComputeRowMajorStrides(
    const dnnl::memory::dims& dims_tf_order) {
  CHECK_GT(dims_tf_order.size(), 0);
  dnnl::memory::dims strides(dims_tf_order.size(), 1);
  for (int d = strides.size() - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * dims_tf_order[d + 1];
  }
  return strides;
}

dnnl::engine FindOrCreateEngine(::sycl::queue* stream) {
  static auto* stream_engine_map =
      new absl::flat_hash_map<::sycl::queue*, dnnl::engine>;

  {
    absl::ReaderMutexLock lock(&engine_map_mutex);
    if (auto it = stream_engine_map->find(stream);
        it != stream_engine_map->end()) {
      return it->second;
    }
  }

  absl::MutexLock lock(&engine_map_mutex);
  auto [it, inserted] = stream_engine_map->try_emplace(
      stream, dnnl::sycl_interop::make_engine(stream->get_device(),
                                              stream->get_context()));
  return it->second;
}

dnnl::fpmath_mode GetFP32MathMode() {
  // TODO (intel-tf): Add support to check for GPU capability for FP32,
  // TF32 and BF32 math mode.
  return dnnl::fpmath_mode::strict;
}

dnnl::memory CreateDnnlMemory(const dnnl::memory::desc& md,
                              const dnnl::engine& engine, void* data_handle) {
  // TODO(intel-tf): Return absl::StatusOr
  CHECK(engine.get_kind() == dnnl::engine::kind::gpu);
  auto kind = dnnl::sycl_interop::memory_kind::usm;
  if (data_handle == nullptr) {
    return dnnl::sycl_interop::make_memory(md, engine, kind,
                                           DNNL_MEMORY_ALLOCATE);
  } else {
    return dnnl::sycl_interop::make_memory(md, engine, kind, data_handle);
  }
}

}  // namespace sycl
}  // namespace stream_executor
