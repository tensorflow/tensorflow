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
#include "tensorflow/core/tpu/kernels/tpu_compile_op_common.h"

#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"
#include "tensorflow/core/protobuf/tpu/compilation_result.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/protobuf/tpu/dynamic_padding.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry_unloader.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_metrics.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_options.h"
#include "tensorflow/core/tpu/kernels/tpu_fingerprint_lookup.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_compile_interface.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace {

std::string TruncateMessage(const std::string& msg, size_t max_len) {
  if (msg.size() > max_len) {
    return absl::StrCat(msg.substr(0, max_len), " ... [truncated]");
  } else {
    return msg;
  }
}

}  // namespace

namespace tensorflow {
namespace tpu {

CompileOpImplFactory* CompileOpImplFactory::factory_ = nullptr;

/* static */
CompileOpImplFactory* CompileOpImplFactory::Get() { return factory_; }

/* static */
void CompileOpImplFactory::Register(CompileOpImplFactory* factory) {
  CHECK_EQ(factory_, nullptr)
      << "CompileOpImplFactory can only be registered "
         "once and there can only be one factory active and used.";
  factory_ = factory;
}

/* static */ void TpuCompileOpKernelCommon::ExitCountdown(
    Env* env, std::shared_ptr<std::atomic<bool>> done) {
  const int kSleepSeconds = 300;
  LOG(INFO) << "TpuCompileOp was cancelled. Sleeping for " << kSleepSeconds
            << " seconds to give time for TPUCompileOp to finished.";
  env->SleepForMicroseconds(kSleepSeconds * 1000000);
  if (done->load()) {
    // If the TpuCompileOp has finished, then terminate peacefully.
    return;
  }

  LOG(ERROR) << "Aborting process due to cancelled TpuCompileOp. This "
             << "termination is to ensure a consistent state.";
  std::exit(42);
}

/* static */ Status TpuCompileOpKernelCommon::GetDynamicShapes(
    OpKernelContext* ctx, std::vector<TensorShape>* shapes) {
  OpInputList dynamic_shapes;
  TF_RETURN_IF_ERROR(ctx->input_list("dynamic_shapes", &dynamic_shapes));

  shapes->resize(dynamic_shapes.size());
  for (int i = 0; i < dynamic_shapes.size(); ++i) {
    TF_RETURN_IF_ERROR(
        tpu::ShapeTensorToTensorShape(dynamic_shapes[i], &(*shapes)[i]));
  }
  return OkStatus();
}

void TpuCompileOpKernelCommon::Compute(OpKernelContext* ctx) {
  VLOG(1) << "Cloud TPU: TpuCompileOpKernelCommon::Compute";

  std::shared_ptr<std::atomic<bool>> done(new std::atomic<bool>(false));

  CancellationToken token =
      ctx->cancellation_manager()->get_cancellation_token();
  const bool already_cancelled =
      !ctx->cancellation_manager()->RegisterCallback(token, [ctx, done]() {
        if (OpsApiFn()->TpuCompile_ShouldTpuCompileOpIgnoreCancellationFn()) {
          return;
        }

        // Sleep and exit in another thread so the cancellation manager can
        // continue running callbacks.
        Env* env = ctx->env();
        env->SchedClosure([env, done]() { ExitCountdown(env, done); });
      });

  // If the RPC was cancelled before we registered the cancellation callback,
  // don't compile the TPU program.
  OP_REQUIRES(ctx, !already_cancelled,
              errors::Cancelled("RPC cancelled, not compiling TPU program"));

  // We only want to abort the process if a cancellation actually occurs during
  // compilation; we must deregister the callback in the success case. It
  // doesn't hurt to also deregister the callback in the failure case; the
  // CancellationManager ensures that already-registered callbacks will be run
  // once cancellation has started.
  auto cancellation_cleanup = absl::MakeCleanup([ctx, token, done] {
    ctx->cancellation_manager()->DeregisterCallback(token);
    done->store(true);
  });

  Status compile_status = ComputeInternal(ctx);
  string status_payload;
  // Construct payload if compile_status is not ok and there's no payload for
  // compilation yet.
  if (!compile_status
           .GetPayload(TpuCompileInterface::kTpuCompileErrorPayloadKey)
           .has_value()) {
    tpu::CompilationResultProto proto;
    proto.set_status_code(compile_status.code());
    proto.set_status_error_message(
        TruncateMessage(compile_status.error_message(), 128));
    status_payload = proto.SerializeAsString();
  }
  OP_REQUIRES_OK_OR_SET_PAYLOAD(ctx,
                                TpuCompileInterface::kTpuCompileErrorPayloadKey,
                                status_payload, compile_status);
}

Status TpuCompileOpKernelCommon::CompileLocallyAndFillHostCache(
    FunctionLibraryRuntime* flib_runtime,
    const SessionMetadata* session_metadata,
    const TpuMeshStateInterface* mesh_state,
    const std::vector<TensorShape>& dynamic_shapes,
    const OpInputList& guaranteed_constants, const TpuCompilationCacheKey& key,
    TpuProgramGroupInterface* tpu_program_group) {
  Status status = CompileLocallyAndFillHostCacheInternal(
      flib_runtime, session_metadata, mesh_state, dynamic_shapes,
      guaranteed_constants, key, tpu_program_group);
  OkOrSetErrorCounterPayload(
      tensorflow::core::platform::ErrorSourceProto::TPU_COMPILE_OP, status);
  return status;
}

Status TpuCompileOpKernelCommon::CompileLocallyAndFillHostCacheInternal(
    FunctionLibraryRuntime* flib_runtime,
    const SessionMetadata* session_metadata,
    const TpuMeshStateInterface* mesh_state,
    const std::vector<TensorShape>& dynamic_shapes,
    const OpInputList& guaranteed_constants, const TpuCompilationCacheKey& key,
    TpuProgramGroupInterface* tpu_program_group) {
  absl::Time start_time = absl::Now();
  std::vector<TensorShape> arg_shapes;
  TF_RETURN_IF_ERROR(
      ComputeArgumentShapes(metadata_, dynamic_shapes, &arg_shapes));
  Status compile_status;
  if (use_mlir_) {
    const ConfigProto* config = flib_runtime->config_proto();
    ConfigProto::Experimental::MlirBridgeRollout rollout_state =
        GetMlirBridgeRolloutState(config ? absl::make_optional(*config)
                                         : absl::nullopt);
    compile_status =
        Compile(MlirToHloArgs{mlir_module_, rollout_state}, mesh_state->data(),
                arg_shapes, &key, tpu_program_group);
  } else {
    compile_status =
        Compile(FunctionToHloArgs{&function_,
                                  flib_runtime->GetFunctionLibraryDefinition(),
                                  flib_runtime->graph_def_version(),
                                  {&guaranteed_constants}},
                mesh_state->data(), arg_shapes, &key, tpu_program_group);
  }

  absl::Time end_time = absl::Now();
  auto duration = end_time - start_time;

  const std::string session_name = SessionNameFromMetadata(session_metadata);
  LOG(INFO) << "Compilation of " << key.prefix << " with session name "
            << session_name << " took " << duration << " and "
            << (compile_status.ok() ? "succeeded" : "failed");
  tpu_program_group->LogProgramMemorySummary();
  metrics::UpdateTpuErrorCounter("TpuCompileOp",
                                 error_name(compile_status.code()));
  metrics::UpdateXlaCompilationTime(absl::ToInt64Microseconds(duration));
  TpuCompilationMetrics::IncrementCompilationCount(session_name);

  return compile_status;
}

Status TpuCompileOpKernelCommon::ComputeInternal(OpKernelContext* ctx) {
  VLOG(1) << "Retrieving mesh state";
  // Retrieve the topology from the resource manager
  ResourceMgr* rm = GetTPUConfigResourceMgr();

  TpuMeshStateInterface* mesh_state;
  TF_RETURN_IF_ERROR(rm->Lookup(rm->default_container(),
                                kTpuMeshStateInterfaceResourceName,
                                &mesh_state));
  core::ScopedUnref mesh_state_unref(mesh_state);

  std::vector<TensorShape> dynamic_shapes;
  TF_RETURN_IF_ERROR(GetDynamicShapes(ctx, &dynamic_shapes));

  OpInputList guaranteed_constants;
  // TODO(ycao): Decide whether/how to support guaranteed constants in
  // MLIR-based TF-Compiler Bridge.
  if (!use_mlir_) {
    TF_RETURN_IF_ERROR(
        ctx->input_list("guaranteed_constants", &guaranteed_constants));
  }

  ResourceMgr* resource_mgr = ctx->resource_manager();

  // The session_id needs to be unique among live sessions.
  // Recycled session_id is acceptable if it is unique among live sessions.
  uint64_t session_id = reinterpret_cast<uint64_t>(resource_mgr);
  const TpuCompilationCacheKey key = CreateCompilationCacheKey(
      function_.name(), metadata_.function_library_fingerprint(),
      mlir_module_fingerprint_, guaranteed_constants, dynamic_shapes, metadata_,
      *mesh_state, session_id, resource_mgr);

  // Process-wide cache of TPU executables.
  TpuCompilationCacheInterface* cache;
  TF_RETURN_IF_ERROR(rm->Lookup<TpuCompilationCacheInterface>(
      rm->default_container(), kCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  // Per-step object that ensures that compilation cache entries aren't
  // evicted until the step completes. This mechanism ensures that the
  // downstream TPUExecute Ops in this step will be able to look up the
  // compiled executable even if it is marked for eviction before the step
  // ends.
  //
  // We can't use GetTPUConfigResourceMgr here because it may return the
  // global ResourceMgr, which is not associated with any device, and
  // GraphMgr's ScopedStepContainer only searches ResourceMgrs associated
  // with devices when deleting resources at step boundaries.
  CompilationRefHolder* ref_holder;
  if (ctx->step_container() == nullptr) {
    return errors::FailedPrecondition(
        "TPUCompileOp requires a step container.");
  }
  TF_RETURN_IF_ERROR(
      ctx->step_container()->LookupOrCreate<CompilationRefHolder>(
          ctx->resource_manager(), "ref_holder", &ref_holder,
          [cache](CompilationRefHolder** h) {
            *h = cache->MakePerStepRefHolder();
            return OkStatus();
          }));
  core::ScopedUnref ref_holder_unref(ref_holder);

  int64_t uid;
  std::vector<std::string> proto_key;
  std::vector<std::string> sharding_key;
  std::vector<bool> may_modify_variables;
  absl::Span<const xla::HloProto* const> hlo_metadatas;
  Status status = cache->CompileIfKeyAbsent(
      key, ctx->session_metadata(), ref_holder, &uid, &proto_key, &sharding_key,
      &may_modify_variables, &hlo_metadatas,
      [&](TpuProgramGroupInterface* tpu_program_group) {
        VLOG(1) << "Cloud TPU: Compiling TPU program";
        // When this compile function is invoked, we know that host-memory
        // cache TpuCompilationCache saw a cache miss. There are two codepaths:
        // 1. If persistent cache is disabled, compile locally and populate
        //    host-memory cache.
        // 2. If persistent cache is enabled, we do an additional lookup on
        //    the persistent cache.
        //    - If persistent cache also sees a cache miss, trigger
        //      compilation. Then, populate both persistent cache and
        //      host-memory cache.
        //    - If persistent cache sees a cache hit, retrieve cache entry from
        //      persistent cache to populate host-memory cache without
        //      recompilation. If retrieval failed, compile locally as a
        //      fallback and use the local compilation result to populate
        //      host-memory cache.
        if (persistent_cache_ == nullptr) {
          VLOG(1) << "Persistent compilation cache not enabled. Compiling "
                     "TPU executable locally and populating host-memory cache.";
          return CompileLocallyAndFillHostCache(
              ctx->function_library(), ctx->session_metadata(), mesh_state,
              dynamic_shapes, guaranteed_constants, key, tpu_program_group);
        }
        return LookupPersistentCompilationCacheAndFillCaches(
            ctx->function_library(), ctx->session_metadata(), mesh_state,
            dynamic_shapes, guaranteed_constants, persistent_cache_.get(), key,
            tpu_program_group);
      });

  // `ref_holder` is provided to CompileIfKeyAbsent to ensure that cache
  // entry does not get evicted before TpuExecuteOp runs it and discards
  // `ref_holder`. When TpuCompilationCacheEntryUnloader get destroyed in the
  // event that user closes the session while there are in-flight program
  // executions, it will discard the cache's reference to the cache entry
  // and but not removed the entry until `ref_holder` discards the last
  // reference to the entry. This ensures that the guarantees of
  // `ref_holder` is not violated when this flag is true.
  if (unload_cache_entry_on_session_close_) {
    // Place `unloader` in TPU_SYSTEM device resource manager. Note that
    // - TPUConfigResourceMgr returned by GetTPUConfigResourceMgr() is a special
    //   process-global ResourceMgr. There is only one TPUConfigResourceMgr, and
    //   it is never destroyed.
    // - TPU_SYSTEM device resource manager is a normal device ResourceMgr for
    //   TPU_SYSTEM device. If DirectSession or isolate_session_state are used,
    //   there's one TPU_SYSTEM ResourceMgr for each session, and the
    //   ResourceMgrs will be destroyed when their corresponding session is
    //   closed. Otherwise there's one TPU_SYSTEM ResourceMgr that's only
    //   destroyed when the master-session is destroyed, not when the worker
    //   sessions are destroyed
    TpuCompilationCacheEntryUnloader* unloader;
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()
            ->LookupOrCreate<TpuCompilationCacheEntryUnloader>(
                ctx->resource_manager()->default_container(),
                kCompilationCacheUnloaderResourceName, &unloader,
                [cache](TpuCompilationCacheEntryUnloader** new_unloader) {
                  *new_unloader = new TpuCompilationCacheEntryUnloader(cache);
                  return OkStatus();
                }));
    // Note that LookupOrCreate puts two refcounts on unloader.
    core::ScopedUnref unloader_unref(unloader);
    unloader->AddCacheEntryUid(uid);
  }

  int64_t num_cores_with_compiled_programs = proto_key.size();
  if (proto_key.size() == 1) {
    // SPMD produces 1 program for all cores.
    num_cores_with_compiled_programs = metadata_.num_cores_per_replica();
    if (may_modify_variables.size() == 1) {
      may_modify_variables.resize(metadata_.num_cores_per_replica(),
                                  may_modify_variables[0]);
    }
  }
  if (status.ok() &&
      num_cores_with_compiled_programs +
              (may_modify_variables.size() * static_cast<int>(!use_mlir_)) !=
          ctx->num_outputs() - 1) {
    status = errors::Internal(
        "Number of cores with compiled programs (",
        num_cores_with_compiled_programs, ") + variable states (",
        may_modify_variables.size() * static_cast<int>(!use_mlir_),
        ") + compilation status output != number of compile op outputs (",
        ctx->num_outputs(), ")");
  }

  // TODO(jpienaar): status is not just due to the compilation. At this
  // point we should be failing the execution of the op in some cases and
  // returning a compilation error in others. For now, uniformly return an
  // error and fail in _TPUExecute if status failed here.

  // TODO(misard) the frame id will be wrong if this is ever called from
  // within a function. Consider whether to use the same hack as is
  // present in the rendezvous manager where the function call frame is
  // cast to a uint64, or do something better all around.
  std::string rendezvous_key_base = strings::StrCat(
      "host_compute_rendezvous:", ctx->op_kernel().name(), ":",
      ctx->frame_iter().frame_id, ":", ctx->frame_iter().iter_id, ":");

  // Return compilation status.
  if (!status.GetPayload(TpuCompileInterface::kTpuCompileErrorPayloadKey)
           .has_value()) {
    Tensor output(DT_STRING, TensorShape({}));
    tpu::CompilationResultProto proto;
    proto.set_status_code(status.code());
    if (!status.ok()) {
      proto.set_status_error_message(TruncateMessage(
          absl::StrCat("Compilation failure: ", status.error_message()), 128));
    }
    if (return_hlo_protos_) {
      // Return the HloProtos as part of compilation status.
      for (const xla::HloProto* hlo_metadata : hlo_metadatas) {
        xla::HloProto* hlo_proto = proto.add_hlo_protos();
        *hlo_proto = *hlo_metadata;
      }
    }
    SerializeToTString(proto, &output.scalar<tstring>()());
    ctx->set_output(0, output);
    status.SetPayload(TpuCompileInterface::kTpuCompileErrorPayloadKey,
                      absl::Cord(output.scalar<tstring>()()));
  }

  if (status.ok()) {
    for (int i = 0; i < num_cores_with_compiled_programs; ++i) {
      Tensor output(DT_STRING, TensorShape({3}));
      if (proto_key.size() == 1) {
        output.vec<tstring>()(0) = proto_key[0];
      } else {
        output.vec<tstring>()(0) = proto_key[i];
      }
      output.vec<tstring>()(1) = rendezvous_key_base;
      if (sharding_key.empty()) {
        output.vec<tstring>()(2) = "";
      } else if (sharding_key.size() == 1) {
        output.vec<tstring>()(2) = sharding_key[0];
      } else {
        TF_RET_CHECK(sharding_key.size() == num_cores_with_compiled_programs);
        output.vec<tstring>()(2) = sharding_key[i];
      }
      ctx->set_output(i + 1, output);
    }
    if (!use_mlir_) {
      // If any of the programs may modify a variable, then return that all
      // do as the only current state being tracked here is if a model is
      // read-only or not.
      bool may_modify = false;
      for (bool m : may_modify_variables) {
        may_modify = may_modify || m;
      }
      for (int i = 0; i < may_modify_variables.size(); ++i) {
        Tensor output(DT_BOOL, TensorShape({}));
        output.scalar<bool>()() = may_modify;
        ctx->set_output(i + num_cores_with_compiled_programs + 1, output);
      }
    }
    VLOG(1) << "Cloud TPU: Compilation succeeded";
  } else {
    // Return error in the invalid case.
    for (int i = 0; i < num_computations_; ++i) {
      Tensor output(DT_STRING, TensorShape({3}));
      output.vec<tstring>()(0) = "<<NO PROGRAM AS COMPILATION FAILED>>";
      output.vec<tstring>()(1) = "<<NO RENDEZVOUS KEY AS COMPILATION FAILED>>";
      output.vec<tstring>()(2) = "<<NO SHARDing KEY AS COMPILATION FAILED>>";
      ctx->set_output(i + 1, output);
    }
    if (!use_mlir_) {
      // The TPUCompileMLIR op does not have MayModifyVariable output
      for (int i = 0; i < num_computations_; ++i) {
        Tensor output(false);
        ctx->set_output(i + num_computations_ + 1, output);
      }
    }
  }
  return status;
}

Status TpuCompileOpKernelCommon::RegisterXLAFingerprints(
    const std::vector<TensorShape>& arg_shapes,
    TpuProgramGroupInterface* tpu_program_group, uint64 fingerprint) {
  // TODO(chiachenc): Support only one program for now.
  if (tpu_program_group->program_count() != 1) {
    LOG(INFO) << "Found " << tpu_program_group->program_count()
              << " programs. Skip fingerprint registration.";
  } else {
    ResourceMgr* rm = GetTPUConfigResourceMgr();
    tpu::TpuFingerprintLookup* fingerprint_lookup;
    TF_RETURN_IF_ERROR(rm->LookupOrCreate<tpu::TpuFingerprintLookup>(
        rm->default_container(), tpu::kFingerprintLookupResourceName,
        &fingerprint_lookup, [&](tpu::TpuFingerprintLookup** new_lookup) {
          *new_lookup = tpu::TpuFingerprintLookup::Create();
          return OkStatus();
        }));
    uint64 tf_fingerprint =
        tpu::CreateFingerprintWithNameAndShapes(fingerprint, arg_shapes);
    std::string xla_fingerprint = tpu_program_group->fingerprint(0);
    VLOG(1) << "Registering TF fingerprint: " << tf_fingerprint
            << " with XLA fingerprint: " << xla_fingerprint;
    fingerprint_lookup->RegisterIntermediateAndValuePair(
        tf_fingerprint, std::move(xla_fingerprint));
  }
  return OkStatus();
}
}  // namespace tpu
}  // namespace tensorflow
