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
#include "tensorflow/core/tpu/kernels/tpu_program_group.h"

#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

namespace tensorflow {
namespace tpu {
namespace {
namespace se_tpu = ::stream_executor::tpu;
using stream_executor::port::Status;
}  // namespace

TPUExecutableInfoProto TpuProgramGroup::ConstructExecutableInfo(
    const XLA_TpuProgram* xla_tpu_program) {
  VLOG(1) << "ConstructExecutableInfo";
  TpuSerializedProto serialized_executable_info = {};
  StatusHelper status;
  OpsApiFn()->TpuProgram_GetExecutableInfoFn(
      xla_tpu_program, &serialized_executable_info, status.c_status);
  TPUExecutableInfoProto executable_info;
  if (status.ok()) {
    executable_info = se_tpu::DeserializeProto<TPUExecutableInfoProto>(
        serialized_executable_info);
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_executable_info);
  }
  return executable_info;
}

TPUHostTransferInfoProto TpuProgramGroup::ConstructHostTransferInfo(
    const XLA_TpuProgram* xla_tpu_program) {
  VLOG(1) << "ConstructHostTransferInfo";
  TpuSerializedProto serialized_host_transfer_info = {};
  StatusHelper status;
  OpsApiFn()->TpuProgram_GetHostTransferInfoFn(
      xla_tpu_program, &serialized_host_transfer_info, status.c_status);
  TPUHostTransferInfoProto host_transfer_info;
  if (status.ok()) {
    host_transfer_info = se_tpu::DeserializeProto<TPUHostTransferInfoProto>(
        serialized_host_transfer_info);
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_host_transfer_info);
  }
  return host_transfer_info;
}

xla::HloProto TpuProgramGroup::ConstructHloMetadata(
    const XLA_TpuProgram* xla_tpu_program) {
  VLOG(1) << "ConstructHloMetadata";
  TpuSerializedProto serialized_hlo_metadata = {};
  StatusHelper status;
  OpsApiFn()->TpuProgram_GetHloMetadataFn(
      xla_tpu_program, &serialized_hlo_metadata, status.c_status);
  xla::HloProto hlo_metadata;
  if (status.ok()) {
    hlo_metadata =
        se_tpu::DeserializeProto<xla::HloProto>(serialized_hlo_metadata);
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_hlo_metadata);
  }
  return hlo_metadata;
}

void TpuProgramGroup::Initialize(
    absl::Span<XLA_TpuProgram* const> xla_tpu_programs) {
  CHECK_GT(xla_tpu_programs.size(), 0);
  CHECK_EQ(program_count(), 0) << "Reinitialization of an existing "
                                  "`TpuProgramGroup` instance is prohibited.";
  set_tpu_programs(xla_tpu_programs);

  CHECK_EQ(tpu_program_fingerprints_.size(), 0);
  set_fingerprints();

  std::vector<bool> may_modify_variables_array(tpu_programs_.size(), false);
  std::vector<TPUExecutableInfoProto> executable_infos(tpu_programs_.size());
  std::vector<TPUHostTransferInfoProto> host_transfer_infos(
      tpu_programs_.size());
  std::vector<xla::HloProto> hlo_metadatas(tpu_programs_.size());
  for (size_t i = 0; i < tpu_programs_.size(); ++i) {
    const XLA_TpuProgram* xla_tpu_program = tpu_programs_[i];
    bool may_modify_variables;
    OpsApiFn()->TpuProgram_GetMayModifyVariablesFn(xla_tpu_program,
                                                   &may_modify_variables);
    may_modify_variables_array[i] = may_modify_variables;
    executable_infos[i] = ConstructExecutableInfo(xla_tpu_program);
    host_transfer_infos[i] = ConstructHostTransferInfo(xla_tpu_program);
    hlo_metadatas[i] = ConstructHloMetadata(xla_tpu_program);
  }

  may_modify_variables_ = may_modify_variables_array;
  executable_infos_ = executable_infos;
  host_transfer_infos_ = host_transfer_infos;
  hlo_metadatas_ = hlo_metadatas;
  RefreshHloMetadatasPtrs();
}

bool TpuProgramGroup::has_sharding_program() const {
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    if (!OpsApiFn()->TpuProgram_HasShardingFn(tpu_program)) {
      return false;
    }
  }
  return true;
}

size_t TpuProgramGroup::program_count() const { return tpu_programs_.size(); }

int64_t TpuProgramGroup::program_size() const {
  int64_t total_size = 0;
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    total_size += OpsApiFn()->TpuProgram_GetProgramSizeFn(tpu_program);
  }
  return total_size;
}

bool TpuProgramGroup::LogProgramMemorySummary() {
  bool success = true;
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    success &= OpsApiFn()->TpuProgram_LogProgramMemorySummaryFn(tpu_program);
  }
  return success;
}

void TpuProgramGroup::UnloadAndDestroyPrograms() {
  for (XLA_TpuProgram* tpu_program : tpu_programs_) {
    StatusHelper status;
    OpsApiFn()->TpuProgram_UnloadAndDestroyFn(tpu_program, status.c_status);
    auto s = status.status();
    if (!s.ok()) {
      LOG(ERROR) << "TpuProgramGroup::UnloadPrograms(): " << s.ToString();
    }
  }
  tpu_programs_.clear();
}

TpuProgramGroup::TpuProgramGroup(TpuProgramGroup&& other)
    : may_modify_variables_(std::move(other.may_modify_variables_)),
      tpu_programs_(std::move(other.tpu_programs_)),
      executable_infos_(std::move(other.executable_infos_)),
      host_transfer_infos_(std::move(other.host_transfer_infos_)),
      hlo_metadatas_(std::move(other.hlo_metadatas_)) {
  RefreshHloMetadatasPtrs();
}

void TpuProgramGroup::set_hlo_metadatas(
    absl::Span<const xla::HloProto> hlo_metadatas) {
  hlo_metadatas_.resize(hlo_metadatas.size());
  for (size_t i = 0; i < hlo_metadatas.size(); ++i) {
    hlo_metadatas_[i] = hlo_metadatas[i];
  }
  RefreshHloMetadatasPtrs();
}

absl::Span<const xla::HloProto* const> TpuProgramGroup::hlo_metadatas() const {
  return hlo_metadatas_ptrs_;
}

const xla::HloProto* TpuProgramGroup::hlo_metadata(int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, hlo_metadatas_ptrs_.size());
  return hlo_metadatas_ptrs_[index];
}

void TpuProgramGroup::RefreshHloMetadatasPtrs() {
  hlo_metadatas_ptrs_.reserve(hlo_metadatas_.size());
  for (const auto& hlo_metadata_internal_ : hlo_metadatas_) {
    hlo_metadatas_ptrs_.push_back(&hlo_metadata_internal_);
  }
}

Status TpuProgramGroup::LogCompilationStats(const TpuCompilationCacheKey& key,
                                            absl::Duration duration) {
  // A placeholder for tracking compilation statistics for future work. The
  // implementation can be pushing into some external storage for analytics.
  return Status::OK();
}

const std::vector<bool>& TpuProgramGroup::may_modify_variables_list() const {
  return may_modify_variables_;
}

void TpuProgramGroup::set_may_modify_variables(
    const std::vector<bool>& may_modify_variables) {
  may_modify_variables_ = may_modify_variables;
}

bool TpuProgramGroup::may_modify_variables(int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  bool may_modify_variables;
  OpsApiFn()->TpuProgram_GetMayModifyVariablesFn(tpu_programs_[index],
                                                 &may_modify_variables);
  return may_modify_variables;
}

const std::vector<XLA_TpuProgram*>& TpuProgramGroup::tpu_programs() const {
  return tpu_programs_;
}

const std::vector<std::string>& TpuProgramGroup::fingerprints() const {
  return tpu_program_fingerprints_;
}

void TpuProgramGroup::set_fingerprints() {
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    TpuProgramFingerprint fingerprint =
        OpsApiFn()->TpuProgram_GetFingerprintFn(tpu_program);
    tpu_program_fingerprints_.emplace_back(
        std::string(fingerprint.bytes, fingerprint.size));
    OpsApiFn()->TpuProgram_DestroyFingerprintFn(fingerprint);
  }
}

const XLA_TpuProgram* TpuProgramGroup::tpu_program(int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  return tpu_programs_[index];
}

void TpuProgramGroup::set_tpu_programs(
    absl::Span<XLA_TpuProgram* const> tpu_programs) {
  tpu_programs_.resize(tpu_programs.size());
  for (size_t i = 0; i < tpu_programs.size(); ++i) {
    tpu_programs_[i] = tpu_programs[i];
  }
}

const TPUExecutableInfoProto& TpuProgramGroup::executable_info(
    int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, executable_infos_.size());
  return executable_infos_[index];
}

const TPUHostTransferInfoProto& TpuProgramGroup::host_transfer_info(
    int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, host_transfer_infos_.size());
  return host_transfer_infos_[index];
}

/*static*/
Status TpuProgramGroup::CompileAndBuild(
    const TpuCompilationRequestProto& compilation_request,
    const XLA_TpuMeshState* mesh_state,
    TpuProgramGroupInterface* tpu_program_group_interface) {
  se_tpu::SerializedProto serialized_compilation_request =
      se_tpu::SerializeProto(compilation_request);
  auto cleanup = gtl::MakeCleanup([serialized_compilation_request] {
    se_tpu::SerializedProto_Free(serialized_compilation_request);
  });
  size_t count = 0;
  XLA_TpuProgram** xla_tpu_programs = nullptr;
  StatusHelper status;
  OpsApiFn()->TpuCompile_CompileAndBuildFn(serialized_compilation_request,
                                           mesh_state, &xla_tpu_programs,
                                           &count, status.c_status);
  if (!status.ok()) {
    VLOG(1) << "Run CompileAndBuild failed.";
    return status.status();
  }

  // SPMD could return 1 result for all partitions.
  TF_RET_CHECK(count == 1 ||
               count == compilation_request.metadata().num_cores_per_replica());

  VLOG(1) << "Initialize TpuProgramGroup.";
  TpuProgramGroup* tpu_program_group =
      tensorflow::down_cast<TpuProgramGroup*>(tpu_program_group_interface);
  tpu_program_group->Initialize(
      absl::MakeConstSpan(&xla_tpu_programs[0], count));
  OpsApiFn()->TpuProgram_FreeArrayFn(xla_tpu_programs);
  return status.status();
}

/*static*/
Status TpuProgramGroup::CompileAndBuild(
    const xrt::XLAComputation& xrt_computation_proto,
    const XLA_TpuMeshState* mesh_state,
    TpuProgramGroupInterface* tpu_program_group_interface) {
  se_tpu::SerializedProto serialized_compilation_request =
      se_tpu::SerializeProto(xrt_computation_proto);
  auto cleanup = gtl::MakeCleanup([serialized_compilation_request] {
    se_tpu::SerializedProto_Free(serialized_compilation_request);
  });
  size_t count = 0;
  XLA_TpuProgram** xla_tpu_programs = nullptr;
  StatusHelper status;
  OpsApiFn()->TpuCompile_XrtCompileAndBuildFn(serialized_compilation_request,
                                              mesh_state, &xla_tpu_programs,
                                              &count, status.c_status);
  if (!status.ok()) {
    VLOG(1) << "Run CompileAndBuild failed.";
    return status.status();
  }

  // SPMD could return 1 result for all partitions.
  int num_cores_per_replica =
      xrt_computation_proto.config().num_cores_per_replica()
          ? xrt_computation_proto.config().num_cores_per_replica()
          : 1;
  TF_RET_CHECK(count == 1 || count == num_cores_per_replica);
  VLOG(1) << "Initialize TpuProgramGroup.";
  TpuProgramGroup* tpu_program_group =
      tensorflow::down_cast<TpuProgramGroup*>(tpu_program_group_interface);
  tpu_program_group->Initialize(
      absl::MakeConstSpan(&xla_tpu_programs[0], count));
  OpsApiFn()->TpuProgram_FreeArrayFn(xla_tpu_programs);
  return status.status();
}

std::vector<XLA_TpuProgram*> TpuProgramGroup::tpu_programs(
    TpuProgramShardingType sharding_type) const {
  std::vector<XLA_TpuProgram*> tpu_programs;
  tpu_programs.reserve(tpu_programs_.size());
  for (size_t i = 0; i < tpu_programs_.size(); ++i) {
    if (OpsApiFn()->TpuProgram_HasShardingFn(tpu_programs_[i])) {
      tpu_programs.push_back(OpsApiFn()->TpuProgram_GetTpuProgramFn(
          tpu_programs_[i], sharding_type));
      CHECK_NE(tpu_programs[i], nullptr);
    }
  }
  return tpu_programs;
}

Status TpuProgramGroup::DeserializeFromRpcResponseProtos(
    const std::vector<TpuSerializedProto>& rpc_response_protos) {
  std::vector<XLA_TpuProgram*> tpu_programs;
  tpu_programs.resize(rpc_response_protos.size());

  for (size_t i = 0; i < rpc_response_protos.size(); ++i) {
    StatusHelper status;
    auto* xla_tpu_program = OpsApiFn()->TpuProgram_NewFn();
    OpsApiFn()->TpuProgram_DeserializeFromGetTpuProgramResponseProtoFn(
        rpc_response_protos[i], xla_tpu_program, status.c_status);
    if (!status.status().ok()) {
      OpsApiFn()->TpuProgram_FreeFn(xla_tpu_program);
      return status.status();
    }
    tpu_programs[i] = xla_tpu_program;
  }

  Initialize(tpu_programs);
  return Status::OK();
}

Status TpuProgramGroup::SerializeExecutable(
    int index, TpuExecutableSerializedProto* executable) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  StatusHelper status;
  OpsApiFn()->TpuProgram_SerializeTpuExecutableFn(tpu_programs_[index],
                                                  executable, status.c_status);
  return status.status();
}

Status TpuProgramGroup::SerializeCompilerMetadata(
    int index, CompilerMetadataSerializedProto* compiler_metadata) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, tpu_programs_.size());
  StatusHelper status;
  OpsApiFn()->TpuProgram_SerializeCompilerMetadataFn(
      tpu_programs_[index], compiler_metadata, status.c_status);
  return status.status();
}
}  // namespace tpu
}  // namespace tensorflow
