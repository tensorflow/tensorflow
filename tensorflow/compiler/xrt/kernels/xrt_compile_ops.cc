/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"
#include "tensorflow/compiler/xrt/xrt_device.h"
#include "tensorflow/compiler/xrt/xrt_metrics.h"
#include "tensorflow/compiler/xrt/xrt_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/monitoring/timed.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class XRTCompileOp : public OpKernel {
 public:
  explicit XRTCompileOp(OpKernelConstruction* ctx);
  ~XRTCompileOp() override;
  XRTCompileOp(const XRTCompileOp&) = delete;
  XRTCompileOp& operator=(const XRTCompileOp&) = delete;

  void Compute(OpKernelContext* ctx) override;

 private:
  Status Compile(OpKernelContext* ctx,
                 const xrt::XLAComputation& computation_proto,
                 std::unique_ptr<xla::LocalExecutable>* program);
};

Status CompilationCacheKey(const xrt::XLAComputation& computation,
                           string* key) {
  const size_t size = computation.ByteSizeLong();
  auto serialized = absl::make_unique<char[]>(size);
  TF_RET_CHECK(
      SerializeToBufferDeterministic(computation, serialized.get(), size));
  uint64 fingerprint = Fingerprint64(absl::string_view(serialized.get(), size));
  *key = absl::StrCat(fingerprint);
  return Status::OK();
}

XRTCompileOp::XRTCompileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

Status XRTCompileOp::Compile(OpKernelContext* ctx,
                             const xrt::XLAComputation& computation_proto,
                             std::unique_ptr<xla::LocalExecutable>* program) {
  const xrt::XLAComputationConfig& config = computation_proto.config();

  // The default config value is 0; treat it as 1 for convenience.
  int num_replicas = config.num_replicas() ? config.num_replicas() : 1;
  TF_RET_CHECK(num_replicas == 1);
  int num_cores_per_replica =
      config.num_cores_per_replica() ? config.num_cores_per_replica() : 1;
  TF_RET_CHECK(num_cores_per_replica == 1);
  TF_RET_CHECK(config.per_core_program_shape_size() == 0);

  // We are guaranteed that the underlying device object won't be deleted out
  // from under us, while the ScopedRef is live.
  class XRTGenericDeviceAccessor::ScopedRef device_ref;
  TF_RETURN_IF_ERROR(XRTGenericDeviceAccessor::InitScopedRef(ctx, &device_ref));

  xla::LocalClient* client = device_ref.client();

  // There is officially no way to use XLA in a client/server architecture where
  // client and server are built from different revisions, because the XLA team
  // does not want to give any guarantees about the stability of the Hlo
  // proto. For cloud TPU this is fine because server and client versions can be
  // assumed to be synced to the same version. For general use the mechanism
  // here (using a snapshot from XlaComputation) works as well as the "official"
  // XLA client/server design, which serializes the same proto between client
  // and server, so in reality is probably fine.
  TF_ASSIGN_OR_RETURN(xla::XlaComputation computation,
                      client->LoadSnapshot(computation_proto.hlo_snapshot()));

  std::vector<xla::Shape> argument_layouts(
      config.program_shape().parameters_size());
  std::vector<const xla::Shape*> argument_layout_ptrs(
      config.program_shape().parameters_size());
  for (int i = 0; i < config.program_shape().parameters_size(); ++i) {
    argument_layouts[i] = xla::Shape(config.program_shape().parameters(i));
    argument_layout_ptrs[i] = &argument_layouts[i];
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(client->default_device_ordinal());
  build_options.set_result_layout(xla::Shape(config.program_shape().result()));
  build_options.set_device_allocator(device_ref.backend()->memory_allocator());
  if (config.has_debug_options()) {
    *build_options.mutable_debug_options() =
        BuildXlaDebugOptions(config.debug_options());
  }

  VLOG(1) << "Building executable";
  TF_ASSIGN_OR_RETURN(
      auto executables,
      client->Compile(computation, argument_layout_ptrs, build_options));
  TF_RET_CHECK(executables.size() == 1);
  *program = std::move(executables[0]);
  return Status::OK();
}

void XRTCompileOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XRTCompileOp::Compute";
  auto timed = monitoring::MakeTimed(xrt_metrics::GetCompileCell());

  ResourceMgr* rm;
  OP_REQUIRES_OK(ctx, XRTGenericDeviceAccessor::GetResourceManager(ctx, &rm));

  const Tensor& computation_input = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(computation_input.shape()),
              errors::Internal("computation input should be a string scalar"));

  xrt::XLAComputation computation_proto;
  OP_REQUIRES(
      ctx,
      computation_proto.ParseFromString(computation_input.scalar<tstring>()()),
      errors::InvalidArgument(
          "Unable to parse computation input to XLAComputation"));

  string key;
  OP_REQUIRES_OK(ctx, CompilationCacheKey(computation_proto, &key));

  // Process-wide cache of XLA executables.
  auto cache_or = GetOrCreateCompilationCache(rm, /*max_number_of_entries=*/0);
  OP_REQUIRES_OK(ctx, cache_or.status());
  auto cache = cache_or.ConsumeValueOrDie();

  int64 uid;
  OP_REQUIRES_OK(
      ctx, cache->CompileIfKeyAbsent(
               key, &uid, [&](std::unique_ptr<xla::LocalExecutable>* program) {
                 VLOG(1) << "Compiling XLA executable";
                 return Compile(ctx, computation_proto, program);
               }));
  std::unique_ptr<XRTCompilationCacheEntryRef> entry;
  OP_REQUIRES_OK(ctx, cache->Lookup(uid, &entry));

  Tensor handle_output(DT_INT64, TensorShape({}));
  handle_output.scalar<int64>()() = uid;
  ctx->set_output(0, handle_output);

  xla::LocalExecutable* executable = entry->get().get_executable();
  xla::ProgramShapeProto program_shape = executable->executable()
                                             ->module()
                                             .config()
                                             .entry_computation_layout()
                                             .ComputeProgramShape()
                                             .ToProto();
  Tensor program_shape_output(DT_STRING, TensorShape({1}));
  program_shape_output.vec<tstring>()(0) = program_shape.SerializeAsString();
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

  ResourceMgr* rm;
  OP_REQUIRES_OK(ctx, XRTGenericDeviceAccessor::GetResourceManager(ctx, &rm));

  // Process-wide cache of XLA executables.
  XRTCompilationCache* cache;
  OP_REQUIRES_OK(ctx, rm->Lookup<XRTCompilationCache>(
                          rm->default_container(),
                          kXRTCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  const Tensor& keys_tensor = ctx->input(0);
  auto flat_keys = keys_tensor.flat<int64>();
  for (int64 i = 0; i < flat_keys.size(); ++i) {
    int64 key = flat_keys(i);
    OP_REQUIRES_OK(ctx, cache->Release(key));
    VLOG(2) << "Released computation handle " << key;
  }
}

}  // namespace

REGISTER_KERNEL_BUILDER(Name("XRTCompile")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("computation")
                            .HostMemory("handle"),
                        XRTCompileOp);
REGISTER_KERNEL_BUILDER(Name("XRTCompile")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("computation")
                            .HostMemory("handle"),
                        XRTCompileOp);

REGISTER_KERNEL_BUILDER(Name("XRTReleaseCompilationHandle")
                            .Device(DEVICE_XLA_CPU)
                            .HostMemory("handle"),
                        XRTReleaseCompilationRefOp);
REGISTER_KERNEL_BUILDER(Name("XRTReleaseCompilationHandle")
                            .Device(DEVICE_XLA_GPU)
                            .HostMemory("handle"),
                        XRTReleaseCompilationRefOp);

}  // namespace tensorflow
