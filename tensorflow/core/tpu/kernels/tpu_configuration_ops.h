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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_CONFIGURATION_OPS_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_CONFIGURATION_OPS_H_

#include <stdint.h>

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

Status CreateTpuCompilationCache(
    ResourceMgr* rmgr, tpu::TpuCompilationCacheInterface** compilation_cache);

xla::StatusOr<std::vector<int32_t>> ConstructDevicesPerHost(
    OpKernelContext* ctx);

// The ConfigureDistributedTpu op is used to start an TPUDriver from
// TensorFlow. It should be run on a TPU_SYSTEM device and returns the
// connection host:port for the CompilationCacheServer. The
// CompilationCacheServer will remain live until the device's Resource Manager
// is cleared or a ShutdownDistributedTpuOp is run on the same device.
class ConfigureDistributedTpuOp : public OpKernel {
 public:
  explicit ConfigureDistributedTpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES(
        ctx, ctx->num_inputs() > 0,
        errors::Internal("_ConfigureDistributedTPU needs at least one input"));
  }
  void Compute(OpKernelContext* ctx) override;
  ~ConfigureDistributedTpuOp() override {}

 private:
  // ConfigureDistributedTpuOp is neither copyable nor movable.
  ConfigureDistributedTpuOp(const ConfigureDistributedTpuOp&) = delete;
  ConfigureDistributedTpuOp& operator=(const ConfigureDistributedTpuOp&) =
      delete;
};

// The WaitForDistributedTpuOp op is used to block execution until
// the distributed Tpu system has started up. It must be run on
// the same TPU_SYSTEM device that ConfigureDistributedTpuOp was run
// on, after all of the InitializeHostForDistributedTpuOp Ops have
// completed.
class WaitForDistributedTpuOp : public OpKernel {
 public:
  explicit WaitForDistributedTpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("startup_timeout_sec", &startup_timeout_sec_));
    OP_REQUIRES(ctx, startup_timeout_sec_ > 0,
                errors::InvalidArgument("startup_timeout_sec ",
                                        startup_timeout_sec_, " must be >0"));
  }
  void Compute(OpKernelContext* ctx) override;
  ~WaitForDistributedTpuOp() override {}

 private:
  // The time to wait for all hosts to start up.
  int startup_timeout_sec_;

  // WaitForDistributedTpuOp is neither copyable nor movable.
  WaitForDistributedTpuOp(const WaitForDistributedTpuOp&) = delete;
  WaitForDistributedTpuOp& operator=(const WaitForDistributedTpuOp&) = delete;
};

// The ShutdownDistributedTpu op is used to stop a running TPUDriver from
// TensorFlow. It should be run on the TPU_SYSTEM device where
// ConfigureDistributedTpuOp was run.
class ShutdownDistributedTpuOp : public OpKernel {
 public:
  explicit ShutdownDistributedTpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;

  ~ShutdownDistributedTpuOp() override {}

 private:
  // ShutdownDistributedTpuOp is neither copyable nor movable.
  ShutdownDistributedTpuOp(const ShutdownDistributedTpuOp&) = delete;
  ShutdownDistributedTpuOp& operator=(const ShutdownDistributedTpuOp&) = delete;
};

// The InitializeHostForDistributedTpu op is used to initialize the
// TPUPlatform on a host in a distributed TPU system. It should be
// run on every host containing TPU devices before any other Ops that use
// TPU are run.
class InitializeHostForDistributedTpuOp : public OpKernel {
 public:
  explicit InitializeHostForDistributedTpuOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    ctx->GetAttr("enable_whole_mesh_compilations",
                 &enable_whole_mesh_compilations_)
        .IgnoreError();
  }

  void Compute(OpKernelContext* ctx) override;

  ~InitializeHostForDistributedTpuOp() override {}

 private:
  // InitializeHostForDistributedTpuOp is neither copyable nor movable.
  InitializeHostForDistributedTpuOp(const InitializeHostForDistributedTpuOp&) =
      delete;
  InitializeHostForDistributedTpuOp& operator=(
      const InitializeHostForDistributedTpuOp&) = delete;

  bool enable_whole_mesh_compilations_ = false;
};

// The SetGlobalTPUArray op is used to initialize the TPUPlatform on a
// host in a distributed TPU system. It should be run on every host
// containing TPU devices before any other Ops that use TPU are run.
class SetGlobalTPUArrayOp : public OpKernel {
 public:
  explicit SetGlobalTPUArrayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;

  ~SetGlobalTPUArrayOp() override {}

 private:
  // SetGlobalTPUArrayOp is neither copyable nor movable.
  SetGlobalTPUArrayOp(const SetGlobalTPUArrayOp&) = delete;
  SetGlobalTPUArrayOp& operator=(const SetGlobalTPUArrayOp&) = delete;
};

// The DisconnectDistributedTpuChips op is used to disconnect all the chips on a
// host from a running TPUDriver instance. It should be run on every host
// containing TPU devices before the ShutdownDistributedTpuOp is run on
// the TPU_SYSTEM.
class DisconnectDistributedTpuChipsOp : public OpKernel {
 public:
  explicit DisconnectDistributedTpuChipsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;

  ~DisconnectDistributedTpuChipsOp() override {}

 private:
  // DisconnectDistributedTpuChipsOp is neither copyable nor movable.
  DisconnectDistributedTpuChipsOp(const DisconnectDistributedTpuChipsOp&) =
      delete;
  DisconnectDistributedTpuChipsOp& operator=(
      const DisconnectDistributedTpuChipsOp&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_CONFIGURATION_OPS_H_
