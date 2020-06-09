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
#ifndef EXPERIMENTAL_BRAIN_TPU_1VM_MINIEXECUTOR_TPU_PROGRAM_H_
#define EXPERIMENTAL_BRAIN_TPU_1VM_MINIEXECUTOR_TPU_PROGRAM_H_

#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {
namespace tpu {

class TpuAotCompilationOptions : public xla::AotCompilationOptions {
 public:
  explicit TpuAotCompilationOptions(int64 replica_count)
      : num_cores_(0), replica_count_(replica_count) {}

  // Returns the ID of the platform to which these options apply.
  se::Platform::Id PlatformId() const override {
    LOG(FATAL) << "Not implemented.";
    return nullptr;
  };

  void set_num_cores(int64 tpu_cores) { num_cores_ = tpu_cores; }
  int64 replica_count() const override { return replica_count_; }
  int64 num_cores() const override { return num_cores_; }

  void set_allow_separate_sharding_programs(bool allow) {
    allow_separate_sharding_programs_ = allow;
  }
  bool allow_separate_sharding_programs() const {
    return allow_separate_sharding_programs_;
  }

  const std::vector<xla::HloModuleConfig::ShardableValueUpdatePair>
  shardable_value_update_pairs() const {
    return shardable_value_update_pairs_;
  }
  void set_shardable_value_update_pairs(
      std::vector<xla::HloModuleConfig::ShardableValueUpdatePair> pairs) {
    shardable_value_update_pairs_ = std::move(pairs);
  }

 private:
  int64 num_cores_;
  int64 replica_count_;

  // Whether to allow the compiler to create separte sharding and unsharding
  // programs, and modify the original program's input/output sharded size. This
  // is used for XLA-chosen sharding on parameters without an on-device loop:
  // the caller can invoke sharding first, then (repeatedly) invoke the sharded
  // main program, and finally invoke the unsharding program when it needs the
  // full output.
  bool allow_separate_sharding_programs_ = false;

  // The list of input/output pairs in the main program that could be sharded.
  std::vector<xla::HloModuleConfig::ShardableValueUpdatePair>
      shardable_value_update_pairs_;
};

// An executable capable of being fed to a TPU device.
class TpuProgram {
 public:
  using Status = ::stream_executor::port::Status;

  virtual ~TpuProgram() = default;

  static Status Build(
      const TPUCompileMetadataProto& metadata,
      const tensorflow::XlaCompiler::CompilationResult& compilation_result,
      const std::vector<ShardingAndIndex>& arg_core_mapping,
      const std::vector<std::vector<xla::Shape>>& per_core_arg_shapes,
      const absl::optional<xla::DeviceAssignment>& xla_device_assignment,
      TpuProgram* tpu_program);

  size_t program_count() const {
    return tpu_programs_.size();
  }

  int64_t program_size() const;

  bool LogProgramMemorySummary();

  void UnloadAndDestroyPrograms();

  const std::vector<bool>& may_modify_variables() const {
    return may_modify_variables_;
  }
  void set_may_modify_variables(const std::vector<bool>& may_modify_variables) {
    may_modify_variables_ = may_modify_variables;
  }

  const tf2xla::HostComputeMetadata& host_compute_metadata() const {
    return host_compute_metadata_;
  }
  void set_host_compute_metadata(
      const tf2xla::HostComputeMetadata& host_compute_metadata) {
    host_compute_metadata_ = host_compute_metadata;
  }

  const std::vector<XLA_TpuProgram*>& tpu_programs() const {
    return tpu_programs_;
  }
  void set_tpu_programs(std::vector<XLA_TpuProgram*> tpu_programs) {
    tpu_programs_ = tpu_programs;
  }

  const TPUExecutableInfoProto& executable_info() const {
    return executable_info_;
  }
  void set_executable_info(const TPUExecutableInfoProto& executable_info) {
    executable_info_ = executable_info;
  }

  const TPUHostTransferInfoProto& host_transfer_info() const {
    return host_transfer_info_;
  }
  void set_host_transfer_info(
      const TPUHostTransferInfoProto& host_transfer_info) {
    host_transfer_info_ = host_transfer_info;
  }

  const xla::HloProto& hlo_metadata() const { return hlo_metadata_; }
  void set_hlo_metadata(const xla::HloProto& hlo_metadata) {
    hlo_metadata_ = hlo_metadata;
  }

 private:
  std::vector<bool> may_modify_variables_;
  tf2xla::HostComputeMetadata host_compute_metadata_;

  std::vector<XLA_TpuProgram*> tpu_programs_;  // Not owned.
  TPUExecutableInfoProto executable_info_;
  TPUHostTransferInfoProto host_transfer_info_;
  xla::HloProto hlo_metadata_;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // EXPERIMENTAL_BRAIN_TPU_1VM_MINIEXECUTOR_TPU_PROGRAM_H_
