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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_external.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/compiled_subgraph.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_metrics.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/kernels/trace_util.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace tensorflow {
namespace tpu {

namespace {

int64 get_uid() {
  uint64 unsigned_rand = random::New64() & INT64_MAX;
  return static_cast<int64>(unsigned_rand);
}

void PopulateEntry(const std::string& key, CompiledSubgraph* entry,
                   TpuProgramGroup tpu_program_group) {
  // Make the unique keys for each cached proto.
  for (int i = 0; i < tpu_program_group.program_count(); ++i) {
    entry->proto_key.push_back(ProtoKeyForComputation(key, i));
  }

  entry->tpu_program_group =
      absl::make_unique<TpuProgramGroup>(std::move(tpu_program_group));
  entry->initialized = true;

  if (entry->initialization_status.ok()) {
    // Compute the entries total size once all members are initialized.
    entry->total_size = entry->ComputeTotalSize();
  }
}

std::unique_ptr<CompiledSubgraph> CreateAndInitializeCompiledSubgraph(
    CompiledSubgraph* main_entry) {
  auto entry = absl::make_unique<CompiledSubgraph>();
  entry->main_entry = main_entry;
  entry->tpu_program_group = absl::make_unique<TpuProgramGroup>();
  return entry;
}
}  // namespace

CompiledSubgraph* TpuCompilationCacheExternal::InitializeEntry(
    const string& key,
    const std::function<Status(TpuProgramGroupInterface*)>& initialize_program,
    const TpuCompilationCacheKey& subgraph_key) {
  CompiledSubgraph* main_entry = new CompiledSubgraph();
  main_entry->parent = this;
  main_entry->subgraph_key = key;
  main_entry->uid = get_uid();
  main_entry->cache_entry_debug_string = subgraph_key.prefix;
  VLOG(1) << "Cache Initializing Entry Session Debug "
          << main_entry->cache_entry_debug_string;

  // Add the entry to the cache, with size zero since there are no compiled
  // programs in it. Once the subgraph has been compiled,
  // UpdateEntryAfterCompilation will be called to potentially mark old entries
  // that don't fit any more for eviction.
  //
  // At this point there is one reference to entry, which is owned by the caller
  // who created the entry. A second reference, owned by the cache, will be
  // added below since we leave the entry in the 'marked for eviction' state
  // here.
  InsertEntry(key, main_entry);

  // Initialize the programs outside the lock so that other cache operations
  // can proceed during the (potentially lengthy) initialization.
  Status initialization_status;

  TpuProgramGroup tpu_program_group;
  {
    mu_.Unlock();
    {
      profiler::TraceMe compile_programs_traceme(
          "TPU compilation cache compile",
          /*level=*/2);
      initialization_status = initialize_program(&tpu_program_group);
    }
    mu_.Lock();
  }

  main_entry->initialization_status = initialization_status;

  // Add the entry to the uid index.
  auto uid_inserted = entries_by_uid_.insert(
      std::pair<int64, CompiledSubgraph*>(main_entry->uid, main_entry));
  CHECK(uid_inserted.second);

  if (tpu_program_group.has_sharding_program()) {
    main_entry->sharding_entry =
        CreateAndInitializeCompiledSubgraph(main_entry);
    TpuProgramGroup sharding_programs;
    sharding_programs.Initialize(
        tpu_program_group.tpu_programs(TpuProgramShardingType::kSharding));
    PopulateEntry(key, main_entry->sharding_entry.get(),
                  std::move(sharding_programs));

    main_entry->unsharding_entry =
        CreateAndInitializeCompiledSubgraph(main_entry);
    TpuProgramGroup unsharding_programs;
    unsharding_programs.Initialize(
        tpu_program_group.tpu_programs(TpuProgramShardingType::kUnsharding));
    PopulateEntry(key, main_entry->unsharding_entry.get(),
                  std::move(unsharding_programs));
  }

  PopulateEntry(key, main_entry, std::move(tpu_program_group));

  for (int64 i = 0; i < main_entry->proto_key.size(); ++i) {
    auto entry_inserted = entries_by_proto_key_.insert(
        std::pair<std::string, std::pair<CompiledSubgraph*, int>>(
            main_entry->proto_key[i], std::make_pair(main_entry, i)));
    CHECK(entry_inserted.second);
  }

  // Add the size to marked_for_eviction_size_ since it will be adjusted down
  // again when the newly-created entry gets unmarked.
  marked_for_eviction_size_ += main_entry->total_size;
  return main_entry;
}
}  // namespace tpu
}  // namespace tensorflow
