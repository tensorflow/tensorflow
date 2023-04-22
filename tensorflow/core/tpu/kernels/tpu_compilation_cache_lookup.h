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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_LOOKUP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_LOOKUP_H_

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"

namespace tensorflow {
namespace tpu {

// TODO(b/162241759): consider merging TpuCompilationCacheLookup and
// TpuCompilationCacheInterface.
// Base class allowing Execute Ops to look up TPU programs. Different subclasses
// are used when the execute Op is in the same address space as the compile Op,
// and when they need to communicate over RPC.
class TpuCompilationCacheLookup : public ResourceBase {
 public:
  ~TpuCompilationCacheLookup() override = default;

  // Looks up an executable corresponding to the model-parallel core index of
  // the subgraph represented by key. On success a wrapper for the proto is
  // returned in program. The wrapper is guaranteed to be valid only during the
  // execution of the Op requesting the proto.
  //
  // Only one of the main, sharding, unsharding entries is fetched, as specified
  // in fetch_target.
  //
  // If the compilation does not create sharding/unsharding programs, but the
  // fetch_target requests one of them, then after this call
  //   (*entry)->get().get_executable() will return nullptr.
  virtual Status Lookup(const string& proto_key,
                        std::unique_ptr<CompilationCacheEntryRef>* entry,
                        CompilationCacheFetchTarget fetch_target) = 0;

  virtual Status Lookup(const string& proto_key,
                        std::unique_ptr<CompilationCacheEntryRef>* entry) {
    return Lookup(proto_key, std::move(entry),
                  CompilationCacheFetchTarget::MAIN);
  }

  // Looks up an executable corresponding to the model-parallel core index of
  // the subgraph represented by uid. On success a wrapper for the proto is
  // returned in program. The wrapper is guaranteed to be valid only during the
  // execution of the Op requesting the proto.
  virtual Status Lookup(int64_t uid, int proto_index,
                        std::unique_ptr<CompilationCacheEntryRef>* entry,
                        CompilationCacheFetchTarget fetch_target) = 0;

  virtual Status Lookup(int64_t uid, int proto_index,
                        std::unique_ptr<CompilationCacheEntryRef>* entry) {
    return Lookup(uid, proto_index, std::move(entry),
                  CompilationCacheFetchTarget::MAIN);
  }
};
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_LOOKUP_H_
