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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_GROUP_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_GROUP_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xrt/xrt.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_executable_info.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_mesh_state_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {
namespace tpu {

class TpuAotCompilationOptions : public xla::AotCompilationOptions {
 public:
  explicit TpuAotCompilationOptions(int64_t replica_count)
      : num_cores_(0), replica_count_(replica_count) {}

  // Returns the ID of the platform to which these options apply.
  se::Platform::Id PlatformId() const override {
    LOG(FATAL) << "Not implemented.";
    return nullptr;
  };

  void set_num_cores(int64_t tpu_cores) { num_cores_ = tpu_cores; }
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

class TpuProgramGroup : public TpuProgramGroupInterface {
 public:
  using Status = ::stream_executor::port::Status;

  // Compiles Mlir or TF function computation by lowering into HLO IR and
  // returns TPU programs ready for execution.
  static Status CompileAndBuild(
      const TpuCompilationRequestProto& compilation_request,
      const XLA_TpuMeshState* mesh_state,
      TpuProgramGroupInterface* tpu_program_group_interface);

  // Compiles HLO IR and returns TPU programs ready for execution.
  static Status CompileAndBuild(
      const xrt::XLAComputation& xrt_computation_proto,
      const XLA_TpuMeshState* mesh_state,
      TpuProgramGroupInterface* tpu_program_group_interface);

  // Initializes `TpuProgramGroup` object with `xla_tpu_programs`.
  void Initialize(absl::Span<XLA_TpuProgram* const> xla_tpu_programs);

  TpuProgramGroup() = default;
  TpuProgramGroup(TpuProgramGroup&& other);
  TpuProgramGroup& operator=(TpuProgramGroup&&) = delete;

  bool has_sharding_program() const override;

  size_t program_count() const override;

  int64_t program_size() const override;

  bool LogProgramMemorySummary() override;

  void UnloadAndDestroyPrograms() override;

  Status LogCompilationStats(const TpuCompilationCacheKey& key,
                             absl::Duration duration) override;

  const std::vector<bool>& may_modify_variables_list() const override;
  void set_may_modify_variables(const std::vector<bool>& may_modify_variables);
  bool may_modify_variables(int index) const override;

  const std::vector<std::string>& fingerprints() const;
  void set_fingerprints();

  const std::vector<XLA_TpuProgram*>& tpu_programs() const;
  std::vector<XLA_TpuProgram*> tpu_programs(TpuProgramShardingType type) const;
  const XLA_TpuProgram* tpu_program(int index) const override;
  void set_tpu_programs(absl::Span<XLA_TpuProgram* const> tpu_programs);

  const TPUExecutableInfoProto& executable_info(int index) const override;

  const TPUHostTransferInfoProto& host_transfer_info(int index) const override;
  void set_hlo_metadatas(absl::Span<const xla::HloProto> hlo_metadatas);
  const xla::HloProto* hlo_metadata(int index) const;
  absl::Span<const xla::HloProto* const> hlo_metadatas() const override;

  // Deserializes `GetTpuProgramResponse` protos from remote cache.
  Status DeserializeFromRpcResponseProtos(
      const std::vector<TpuSerializedProto>& rpc_response_protos);

  // Serializes executable proto from the TPU program for the given core
  // `index`.
  Status SerializeExecutable(int index,
                             TpuExecutableSerializedProto* executable) const;

  // Serializes compiler metadata of the TPU program for the given core `index`.
  Status SerializeCompilerMetadata(
      int index, CompilerMetadataSerializedProto* compiler_metadata) const;

  // Serializes host compute metadata of the TPU program for the given core
  // `index`.
  Status SerializeHostComputeMetadata(
      int index,
      HostComputeMetadataSerializedProto* host_compute_metadata) const;

 private:
  TPUExecutableInfoProto ConstructExecutableInfo(
      const XLA_TpuProgram* tpu_program);
  TPUHostTransferInfoProto ConstructHostTransferInfo(
      const XLA_TpuProgram* tpu_program);
  xla::HloProto ConstructHloMetadata(const XLA_TpuProgram* tpu_program);

  // Update `hlo_metadatas__ptrs_` array from `hlo_metadatas_`. This needs to be
  // called on `hlo_metadatas_` change(s).
  void RefreshHloMetadatasPtrs();

  std::vector<bool> may_modify_variables_;
  std::vector<std::string> tpu_program_fingerprints_;

  std::vector<XLA_TpuProgram*> tpu_programs_;  // Not owned.
  std::vector<TPUExecutableInfoProto> executable_infos_;
  std::vector<TPUHostTransferInfoProto> host_transfer_infos_;

  // To be consistent with the TpuProgramGroupInterface::hlo_metadatas()
  // signature, we store HloProto values in hlo_metadatas_ when
  // set_hlo_metadata(...) is called, and return their pointers from
  // hlo_metadatas_ptrs_ when hlo_metadatas() is called. hlo_metadata_ptrs_ is
  // refreshed whenever hlo_metadatas_ is set or the object is moved.
  std::vector<xla::HloProto> hlo_metadatas_;  // Owned.
  std::vector<const xla::HloProto*> hlo_metadatas_ptrs_;

  TF_DISALLOW_COPY_AND_ASSIGN(TpuProgramGroup);
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_PROGRAM_GROUP_H_
