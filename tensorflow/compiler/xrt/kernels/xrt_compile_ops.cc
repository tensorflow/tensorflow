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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

const int kDefaultCacheSize = 100;

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
  string serialized;
  TF_RET_CHECK(SerializeToStringDeterministic(computation, &serialized));
  uint64 fingerprint = Fingerprint64(serialized);
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
  TF_RETURN_IF_ERROR(
      XRTGenericDeviceAccessor::InitScopedRef(ctx, 0, &device_ref));

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

  std::vector<const xla::Shape*> argument_layouts(
      config.program_shape().parameters_size());
  for (int i = 0; i < config.program_shape().parameters_size(); ++i) {
    argument_layouts[i] = &config.program_shape().parameters(i);
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(client->default_device_ordinal());
  build_options.set_result_layout(config.program_shape().result());
  build_options.set_device_allocator(device_ref.backend()->memory_allocator());

  VLOG(1) << "Building executable";
  auto compile_result =
      client->Compile(computation, argument_layouts, build_options);
  if (!compile_result.ok()) {
    return compile_result.status();
  }
  *program = std::move(compile_result.ValueOrDie());
  return Status::OK();
}

void XRTCompileOp::Compute(OpKernelContext* ctx) {
  VLOG(1) << "XRTCompileOp::Compute";

  ResourceMgr* rm;
  OP_REQUIRES_OK(ctx, XRTGenericDeviceAccessor::GetResourceManager(ctx, &rm));

  const Tensor& computation_input = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(computation_input.shape()),
              errors::Internal("computation input should be a string scalar"));

  xrt::XLAComputation computation_proto;
  OP_REQUIRES(
      ctx,
      computation_proto.ParseFromString(computation_input.scalar<string>()()),
      errors::InvalidArgument(
          "Unable to parse computation input to XLAComputation"));

  string key;
  OP_REQUIRES_OK(ctx, CompilationCacheKey(computation_proto, &key));

  // Process-wide cache of XLA executables.
  XRTCompilationCache* cache;
  OP_REQUIRES_OK(ctx,
                 rm->LookupOrCreate<XRTCompilationCache>(
                     rm->default_container(), kXRTCompilationCacheResourceName,
                     &cache, [](XRTCompilationCache** new_cache) {
                       *new_cache = new XRTCompilationCache(kDefaultCacheSize);
                       return Status::OK();
                     }));
  core::ScopedUnref cache_unref(cache);

  int64 uid;
  OP_REQUIRES_OK(
      ctx, cache->CompileIfKeyAbsent(
               key, &uid, [&](std::unique_ptr<xla::LocalExecutable>* program) {
                 VLOG(1) << "Compiling XLA executable";
                 return Compile(ctx, computation_proto, program);
               }));

  Tensor output(DT_INT64, TensorShape({}));
  output.scalar<int64>()() = uid;
  ctx->set_output(0, output);
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

  const Tensor& key_tensor = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(key_tensor.shape()),
              errors::Internal("computation key should be a string scalar"));
  int64 uid = key_tensor.scalar<int64>()();

  ResourceMgr* rm;
  OP_REQUIRES_OK(ctx, XRTGenericDeviceAccessor::GetResourceManager(ctx, &rm));

  // Process-wide cache of XLA executables.
  XRTCompilationCache* cache;
  OP_REQUIRES_OK(ctx, rm->Lookup<XRTCompilationCache>(
                          rm->default_container(),
                          kXRTCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  OP_REQUIRES_OK(ctx, cache->Release(uid));

  VLOG(2) << "Released computation handle " << uid;
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
