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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_COMPILED_SUBGRAPH_H_
#define TENSORFLOW_CORE_TPU_KERNELS_COMPILED_SUBGRAPH_H_

#include <memory>
#include <string>

#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"

namespace tensorflow {
namespace tpu {

// Forward declaration to avoid circular dependency.
class TpuCompilationCacheInterface;

// Cache for compiled TPU program.
//
// Each key identifies a unique subgraph, and the value is the vector of
// protos that are emitted by compiling the subgraph.
//
// When a subgraph is considered for compilation, the client calls
//
// auto subgraph_key = <compute key for subgraph>;
// auto compile_function = <lambda to compile subgraph into protos>;
// auto per_step_ref_holder = <container to control lifetime of cached
// results>;
// int64 uid;
// std::vector<string> proto_key;
// CompileIfKeyAbsent(subgraph_key, per_step_ref_holder, &uid, &proto_key,
//                    compile_function);
//
// where subgraph_key is the key computed for the subgraph. On success,
// proto_key contains a vector of keys, where proto_key[i] can be used to look
// up the ith proto compiled from the subgraph, and uid contains an identifier
// that can be used in place of key for clients that require cheap
// serializable handles. If the compiled protos were not present in the cache,
// compile_function would be called to generate them. per_step_ref_holder
// extends the lifetime of cached results: it is guaranteed that the protos
// indicated in proto_key will be available for lookup for at least as long as
// per_step_ref_holder is not deleted.
//
// If the caller passes nullptr instead of a per_step_ref_holder then the
// caller is responsible for calling Release(subgraph_key) once for every call
// to CompileIfKeyAbsent(subgraph_key, ...) to discard the reference to the
// compilation results, after the caller is sure it will not look up the
// compiled executables again.
//
// Subsequently the client can call
//
// std::unique_ptr<CompilationCacheEntryRef> entry;
// Lookup(proto_key, &entry);
// auto proto = entry->get();
//
// or
//
// std::unique_ptr<CompilationCacheEntryRef> entry;
// Lookup(uid, proto_index, &entry);
// auto proto = entry->get();
//
// to access a cached proto.
// TODO(misard) Switch the existing TPU ops to use uid+proto_index instead of
// string keys for proto lookups.
//
//
// Usage details within the system:
//
// This cache lives in the resource manager of the TPU_SYSTEM device where the
// compiler runs, typically worker 0 of the system. The cache is discarded and
// a new one created whenever the system is reinitialized.
//
// A compiled subgraph is placed into the cache using a key that is a
// combination of the function name, guaranteed_constants, the shapes of the
// dynamic inputs to the subgraph, and the function library in use at the time
// of execution.
//
// Whenever a compile Op is run, it looks to see if there is already an entry
// in the cache corresponding to that Op and the current dynamic shapes, and
// creates one if not. The entry is marked as most recently used in the cache
// by the compile Op. The entry is reference counted. The cache owns one entry
// , and each step that has executed a compile Op referring to the entry owns
// a reference until that step completes.
//
// If the cache exceeds a configured storage limit, entries are marked for
// eviction in order of least recently used. An entry is not evicted until all
// references to it are discarded, so an entry that is marked for eviction can
// still be looked up by the execute Ops in a running step. If another Compile
// Op looks up an entry that is marked for eviction, the entry will be
// unmarked and set to most recently used.
//
struct CompiledSubgraph : public core::RefCounted {
  TpuCompilationCacheInterface* parent = nullptr;  // Not owned.

  bool initialized = false;

  // The Status returned by the compilation function when the entry is
  // initialized. This status will be returned to any client that requests the
  // entry.
  Status initialization_status;

  // Counter to keep track of LRU entries for the eviction policy.
  int64_t last_use = -1;

  // The unique key describing this entry.
  std::string subgraph_key;

  // The uid describing this entry.
  int64_t uid;

  // Compilation cache proto key to identify the cache entry.
  std::vector<std::string> proto_key;

  // Fingerprints of sharding programs if there is any.
  std::vector<std::string> sharding_key;

  // The number of 'external' client-held references to the entry.
  int external_references = 0;

  // The sum of the SpaceUsed of each of the elements of programs; an estimate
  // of how much RAM the entry consumes, used to determine when entries must
  // be marked for eviction.
  int64_t total_size = 0;

  // Debug info in case we miss.
  std::string cache_entry_debug_string;

  // Entries representing the associated sharding and unsharding programs,
  // which share the same life time of the owning main entry, so we always use
  // the main entry's ref count.
  std::unique_ptr<CompiledSubgraph> sharding_entry;
  std::unique_ptr<CompiledSubgraph> unsharding_entry;

  // Only used for the nested sharding/unsharding entries to point to the
  // owning main entry.
  CompiledSubgraph* main_entry = nullptr;

  // Compiled TPU program group.
  std::unique_ptr<TpuProgramGroupInterface> tpu_program_group;

  // Computes total program size.
  size_t ComputeTotalSize() const {
    CHECK_EQ(total_size, 0);
    int64_t size = tpu_program_group->program_size();

    if (sharding_entry != nullptr) {
      size += sharding_entry->total_size;
    }
    if (unsharding_entry != nullptr) {
      size += unsharding_entry->total_size;
    }
    return size;
  }
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_COMPILED_SUBGRAPH_H_
