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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_metrics.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/kernels/trace_util.h"

namespace tensorflow {
namespace tpu {

namespace {

int64 get_uid() {
  uint64 unsigned_rand = random::New64() & INT64_MAX;
  return static_cast<int64>(unsigned_rand);
}

void PopulateEntry(const std::string& key, CompiledSubgraph* entry,
                   TpuProgramGroup& tpu_program_group) {
  // Make the unique keys for each cached proto.
  for (int i = 0; i < tpu_program_group.program_count(); ++i) {
    entry->proto_key.push_back(ProtoKeyForComputation(key, i));
  }

  entry->tpu_program_group =
      absl::make_unique<TpuProgramGroup>(std::move(tpu_program_group));
  entry->initialized = true;
}

// Return fingerprint_in_metadata if it's not empty; otherwise read input tensor
// data to compute the fingerprint.
std::string GuaranteedConstFingerprint(
    const string& fingerprint_in_metadata,
    const OpInputList& guaranteed_constants) {
  if (fingerprint_in_metadata.empty()) {
    uint64_t fingerprint = 0;
    for (const auto& constant : guaranteed_constants) {
      fingerprint = TpuCompile_CreateGuaranteedConstFingerprint(
          fingerprint, constant.tensor_data().data(),
          constant.tensor_data().size());
    }
    return std::to_string(fingerprint);
  } else {
    return fingerprint_in_metadata;
  }
}

std::string CreateShapePrefix(
    const std::vector<tensorflow::TensorShape>& dynamic_shapes) {
  std::string shapes_prefix;
  for (const TensorShape& shape : dynamic_shapes) {
    for (int64 size : shape.dim_sizes()) {
      absl::StrAppend(&shapes_prefix, size, ",");
    }
    absl::StrAppend(&shapes_prefix, ";");
  }
  return shapes_prefix;
}

// Include compilation configurations of the arguments that are not captured
// by the called graph.
std::string CreateConfigPrefix(const TPUCompileMetadataProto& metadata) {
  std::string config_prefix;
  for (const auto& arg : metadata.args()) {
    if (arg.is_same_data_across_replicas()) {
      absl::StrAppend(&config_prefix, ":s");
      // Same.
    } else {
      // Different.
      absl::StrAppend(&config_prefix, ":");
    }
    if (arg.enable_xla_sharding() ==
        tpu::TPUCompileMetadataProto::Arg::ALLOWED) {
      // Enabled.
      absl::StrAppend(&config_prefix, "e");
    }
    if (arg.unrestricted_layout()) {
      // Unrestricted.
      absl::StrAppend(&config_prefix, ":u");
    }
    absl::StrAppend(&config_prefix, ",type(", arg.dtype(), ")");
    if (arg.has_shape()) {
      absl::StrAppend(&config_prefix, ",shape(");
      for (const auto& dim : arg.shape().dim()) {
        absl::StrAppend(&config_prefix, dim.size(), ",");
      }
      absl::StrAppend(&config_prefix, ")");
    }
  }
  return config_prefix;
}

}  // namespace

TpuCompilationCacheExternal::EntryRefImpl::EntryRefImpl(
    TpuCompilationCacheInterface* parent, CompiledSubgraph* entry, int index)
    : CompilationCacheEntryRefImpl<TpuCompilationCacheEntry>(parent, entry,
                                                             index) {}

TpuCompilationCacheEntry TpuCompilationCacheExternal::EntryRefImpl::get() {
  if (entry_ == nullptr) {
    // Create an empty entry if the entry is nullptr. This corresponds to
    // non-existing sharding/unsharding entries.
    return TpuCompilationCacheEntry();
  }
  return TpuCompilationCacheEntry(entry_->tpu_program_group.get(), index_);
}

CompiledSubgraph* TpuCompilationCacheExternal::InitializeEntry(
    const string& key,
    const std::function<Status(TpuProgramGroupInterface*)>& initialize_program,
    const TpuCompilationCacheKey& subgraph_key) {
  CompiledSubgraph* main_entry = new CompiledSubgraph();
  main_entry->parent = this;
  main_entry->subgraph_key = key;
  main_entry->uid = get_uid();
  // TODO(henrytan): implement TpuCompilationCacheKey.debug_string.
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

  if (initialization_status.ok()) {
    // Compute the entries total size once all members are initialized.
    main_entry->total_size = tpu_program_group.program_size();
  }

  // TODO(henrytan): handle sharding/unsharding.
  PopulateEntry(key, main_entry, tpu_program_group);

  for (int64 i = 0; i < main_entry->proto_key.size(); ++i) {
    auto entry_inserted = entries_by_proto_key_.insert(
        std::pair<string, std::pair<CompiledSubgraph*, int>>(
            main_entry->proto_key[i], std::make_pair(main_entry, i)));
    CHECK(entry_inserted.second);
  }

  // Add the size to marked_for_eviction_size_ since it will be adjusted down
  // again when the newly-created entry gets unmarked.
  marked_for_eviction_size_ += main_entry->total_size;
  return main_entry;
}

/*static*/ TpuCompilationCacheKey
TpuCompilationCacheExternal::CreateCompilationCacheKey(
    absl::string_view function_name, uint64 function_library_fingerprint,
    absl::string_view mlir_module,
    const tensorflow::OpInputList& guaranteed_constants,
    const std::vector<tensorflow::TensorShape>& dynamic_shapes,
    const tensorflow::tpu::TPUCompileMetadataProto& metadata,
    const TpuMeshStateInterface& mesh_state) {
  VLOG(1) << "FunctionLibraryFingerprint:" << function_library_fingerprint;
  std::string shapes_prefix = CreateShapePrefix(dynamic_shapes);
  VLOG(1) << "shapes_prefix = " << shapes_prefix;
  std::string config_prefix = CreateConfigPrefix(metadata);
  VLOG(1) << "config_prefix = " << config_prefix;
  std::vector<int32_t> flattened_device_ids;
  if (metadata.has_device_assignment()) {
    for (const auto& device :
         metadata.device_assignment().computation_devices()) {
      flattened_device_ids.insert(flattened_device_ids.end(),
                                  device.replica_device_ids().begin(),
                                  device.replica_device_ids().end());
    }
  }
  // TODO(henrytan): return the debug_string.
  const char* prefix =
      TpuCompile_CreateCompilationCacheKey(CompilationCacheKeyProperty{
          config_prefix.data(),
          shapes_prefix.data(),
          function_name.data(),
          mlir_module.data(),
          flattened_device_ids.data(),
          flattened_device_ids.size(),
          guaranteed_constants.size(),
          function_library_fingerprint,
          metadata.num_cores_per_replica(),
          metadata.num_replicas(),
          mesh_state.data(),
      });
  auto buffer_cleanup = gtl::MakeCleanup([prefix]() { delete[] prefix; });
  TpuCompilationCacheKey key;
  key.prefix = prefix;

  // Guaranteed constants can be different across sessions. Use session_handle
  // and guaranteed_const fingerprint to guarantee no collision.
  if (guaranteed_constants.size() > 0) {
    key.has_guaranteed_const = true;
    key.session_handle = metadata.session_handle();
    // Both `metadata` and `guaranteed_constants` lifetime are captured by
    // reference based on the assumption that these variables lifetime is
    // managed through the `TPUCompileOpKernelImpl` that outlives the
    // lifetime of the compilation cache lookups.
    string fingerprint;
    key.guaranteed_const_fingerprint = [&metadata, &guaranteed_constants,
                                        fingerprint]() mutable {
      if (fingerprint.empty()) {
        fingerprint = GuaranteedConstFingerprint(
            metadata.guaranteed_const_fingerprint(), guaranteed_constants);
      }
      return fingerprint;
    };
  }
  return key;
}
}  // namespace tpu
}  // namespace tensorflow
