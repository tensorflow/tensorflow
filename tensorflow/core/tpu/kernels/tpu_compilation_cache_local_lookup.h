/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_LOCAL_LOOKUP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_LOCAL_LOOKUP_H_

#include <cstdint>
#include <memory>
#include <string>

#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_lookup.h"

namespace tensorflow {
namespace tpu {

// Class for looking up TPU programs when the execute and compile Op are in the
// same address space. The proto is simply looked up in the compilation cache,
// without any serialization taking place.
class TpuCompilationCacheLocalLookup : public TpuCompilationCacheLookup {
 public:
  explicit TpuCompilationCacheLocalLookup(TpuCompilationCacheInterface* cache);
  ~TpuCompilationCacheLocalLookup() override;

  absl::Status Lookup(const std::string& proto_key,
                      std::unique_ptr<CompilationCacheEntryRef>* entry,
                      CompilationCacheFetchTarget fetch_target) override;

  absl::Status Lookup(int64_t uid, int proto_index,
                      std::unique_ptr<CompilationCacheEntryRef>* entry,
                      CompilationCacheFetchTarget fetch_target) override;

  std::string DebugString() const override;

 private:
  // The subgraph compilation cache, in the same process address space where the
  // lookups are happening.
  TpuCompilationCacheInterface* cache_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_LOCAL_LOOKUP_H_
