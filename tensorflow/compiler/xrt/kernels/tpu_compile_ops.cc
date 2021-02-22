/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Classes for compiling XLA computations and managing handles that refer to
// them.

#include <string>
#include <vector>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

class XRTCompileOp : public OpKernel {
 public:
  explicit XRTCompileOp(OpKernelConstruction* ctx);
  ~XRTCompileOp() override;
  XRTCompileOp(const XRTCompileOp&) = delete;
  XRTCompileOp& operator=(const XRTCompileOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  Status Compile(const XLA_TpuMeshState* xla_mesh_state,
                 const xrt::XLAComputation& computation_proto,
                 tensorflow::tpu::TpuProgramGroupInterface* tpu_program_group);
};

XRTCompileOp::XRTCompileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

Status XRTCompileOp::Compile(
    const XLA_TpuMeshState* xla_mesh_state,
    const xrt::XLAComputation& computation_proto,
    tensorflow::tpu::TpuProgramGroupInterface* tpu_program_group) {
  return tensorflow::tpu::TpuProgramGroup::CompileAndBuild(
      computation_proto, xla_mesh_state, tpu_program_group);
}

tpu::TpuCompilationCacheKey CompilationCacheKey(
    const xrt::XLAComputation& computation,
    tensorflow::tpu::TpuMeshStateInterface* mesh_state, int num_replicas,
    int num_cores_per_replica) {
  string computation_serialized;
  CHECK(SerializeToStringDeterministic(computation, &computation_serialized));
  tpu::TPUCompileMetadataProto metadata;
  metadata.set_num_replicas(num_replicas);
  metadata.set_num_cores_per_replica(num_cores_per_replica);
  const tpu::TpuCompilationCacheKey key = CreateCompilationCacheKey(
      "compile", 0, tensorflow::Fingerprint64(computation_serialized), {}, {},
      metadata, *mesh_state);
  return key;
}

void ExitCountdown(Env* env, std::shared_ptr<std::atomic<bool>> done) {
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

void XRTCompileOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XRTCompileOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetCompileCell());

  std::shared_ptr<std::atomic<bool>> done(new std::atomic<bool>(false));
  CancellationToken token =
      ctx->cancellation_manager()->get_cancellation_token();
  const bool already_cancelled =
      !ctx->cancellation_manager()->RegisterCallback(token, [ctx, done]() {
        if (tpu::OpsApiFn()
                ->TpuCompile_ShouldTpuCompileOpIgnoreCancellationFn()) {
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
  auto cancellation_cleanup = xla::MakeCleanup([ctx, token, done] {
    ctx->cancellation_manager()->DeregisterCallback(token);
    done->store(true);
  });

  VLOG(1) << "Retrieving pod state";
  // Retrieve the topology from the resource manager
  ResourceMgr* rm = GetTPUConfigResourceMgr();
  tensorflow::tpu::TpuMeshStateInterface* mesh_state;
  OP_REQUIRES_OK(ctx,
                 rm->Lookup(rm->default_container(),
                            tensorflow::tpu::kTpuMeshStateInterfaceResourceName,
                            &mesh_state));
  core::ScopedUnref mesh_state_unref(mesh_state);

  const Tensor& computation_input = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(computation_input.shape()),
              errors::Internal("computation input should be a string scalar"));

  xrt::XLAComputation computation_proto;
  OP_REQUIRES(
      ctx,
      computation_proto.ParseFromString(computation_input.scalar<tstring>()()),
      errors::InvalidArgument(
          "Unable to parse computation input to XLAComputation"));

  const xrt::XLAComputationConfig& config = computation_proto.config();
  int num_replicas = config.num_replicas() ? config.num_replicas() : 1;
  CHECK_GT(num_replicas, 0);
  int num_cores_per_replica =
      config.num_cores_per_replica() ? config.num_cores_per_replica() : 1;

  const tpu::TpuCompilationCacheKey key = CompilationCacheKey(
      computation_proto, mesh_state, num_replicas, num_cores_per_replica);

  // Process-wide cache of Tpu executables.
  tpu::TpuCompilationCacheInterface* cache;
  OP_REQUIRES_OK(ctx, rm->Lookup<tpu::TpuCompilationCacheInterface>(
                          rm->default_container(),
                          tpu::kCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  int64 uid;
  std::vector<string> proto_key;
  std::vector<string> shard_key;
  std::vector<bool> may_modify_variables;
  absl::Span<const xla::HloProto* const> hlo_metadata;
  OP_REQUIRES_OK(
      ctx, cache->CompileIfKeyAbsent(
               key, /*session_metadata=*/nullptr,
               /*per_step_ref_holder=*/nullptr, &uid, &proto_key, &shard_key,
               &may_modify_variables, &hlo_metadata,
               [&](tpu::TpuProgramGroupInterface* tpu_program_group) {
                 VLOG(1) << "Compiling TPU executable";
                 return Compile(mesh_state->data(), computation_proto,
                                tpu_program_group);
               }));

  Tensor output(DT_INT64, TensorShape({}));
  output.scalar<int64>()() = uid;
  ctx->set_output(0, output);

  Tensor program_shape_output(DT_STRING, TensorShape({num_cores_per_replica}));
  for (int64 i = 0; i < num_cores_per_replica; ++i) {
    xla::ProgramShapeProto program_shape =
        hlo_metadata[i]->hlo_module().host_program_shape();
    program_shape_output.vec<tstring>()(i) = program_shape.SerializeAsString();
  }
  ctx->set_output(1, program_shape_output);
}

XRTCompileOp::~XRTCompileOp() = default;

class XRTReleaseCompilationRefOp : public OpKernel {
 public:
  explicit XRTReleaseCompilationRefOp(OpKernelConstruction* ctx);
  ~XRTReleaseCompilationRefOp() override;
  XRTReleaseCompilationRefOp(const XRTReleaseCompilationRefOp&) = delete;
  XRTReleaseCompilationRefOp& operator=(const XRTReleaseCompilationRefOp&) =
      delete;

  void Compute(OpKernelContext* ctx) override;
};

XRTReleaseCompilationRefOp::XRTReleaseCompilationRefOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {}

XRTReleaseCompilationRefOp::~XRTReleaseCompilationRefOp() = default;

void XRTReleaseCompilationRefOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XRTReleaseCompilationRefOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetReleaseCompilationCell());
  ResourceMgr* rm = GetTPUConfigResourceMgr();
  OP_REQUIRES(ctx, rm != nullptr, errors::Internal("No resource manager."));

  // Process-wide cache of Tpu executables.
  tpu::TpuCompilationCacheInterface* cache;
  OP_REQUIRES_OK(ctx, rm->Lookup<tpu::TpuCompilationCacheInterface>(
                          rm->default_container(),
                          tpu::kCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  const Tensor& keys_tensor = ctx->input(0);
  auto flat_keys = keys_tensor.flat<int64>();
  for (int64 i = 0; i < flat_keys.size(); ++i) {
    int64 key = flat_keys(i);
    OP_REQUIRES_OK(ctx, cache->Release(key));
    VLOG(2) << "Released computation handle " << key;
  }
}

REGISTER_KERNEL_BUILDER(Name("XRTCompile")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("computation")
                            .HostMemory("handle"),
                        XRTCompileOp);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseCompilationHandle")
                            .Device(DEVICE_TPU_NODE)
                            .HostMemory("handle"),
                        XRTReleaseCompilationRefOp);

}  // namespace tensorflow
