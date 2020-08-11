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
#include "tensorflow/core/tpu/kernels/tpu_compile_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_program_c_api.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"

namespace tensorflow {
namespace tpu {

namespace {

namespace se_tpu = ::stream_executor::tpu;

using stream_executor::port::Status;
using stream_executor::port::StatusOr;
using xla::Shape;

StatusOr<std::vector<XLA_TpuProgram*>> CompileAheadOfTime(
    std::unique_ptr<xla::HloModuleGroup> module_group,
    const XlaCompiler::CompilationResult& compilation_result,
    const TPUCompileMetadataProto& metadata,
    const std::vector<std::vector<xla::Shape>>& per_core_arg_shapes,
    const std::vector<std::vector<xla::Shape>>& per_core_output_shapes,
    const std::vector<std::vector<std::pair<int, bool>>>&
        per_core_variable_indices,
    const absl::optional<xla::DeviceAssignment>& device_assignment) {
  VLOG(1) << "Run CompileAheadOfTime.";
  TF_ASSIGN_OR_RETURN(TpuAotCompilationRequestProto aot_request,
                      CreateTpuAotCompilationRequest(
                          *module_group, compilation_result, metadata,
                          per_core_arg_shapes, per_core_output_shapes,
                          per_core_variable_indices, device_assignment));
  se_tpu::SerializedProto serialized_aot_request =
      se_tpu::SerializeProto(aot_request);
  auto cleanup = gtl::MakeCleanup([serialized_aot_request] {
    se_tpu::SerializedProto_Free(serialized_aot_request);
  });

  XLA_TpuProgram** xla_tpu_programs = nullptr;
  size_t count = 0;
  StatusHelper status;
  VLOG(1) << "Run TpuCompile_CompileAheadOfTime.";
  CompileApiFn()->TpuCompile_CompileAheadOfTimeFn(
      serialized_aot_request, &xla_tpu_programs, &count, status.c_status);
  VLOG(1) << "Run CompileAheadOfTime completed.";
  if (!status.status().ok()) {
    return status.status();
  }
  std::vector<XLA_TpuProgram*> tpu_programs(count, nullptr);
  for (size_t i = 0; i < count; ++i) {
    tpu_programs[i] = xla_tpu_programs[i];
  }
  TpuProgramApiFn()->TpuProgram_FreeArrayFn(xla_tpu_programs);
  return tpu_programs;
}

StatusOr<std::vector<XLA_TpuProgram*>> CompileAheadOfTime(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::vector<std::vector<xla::Shape>>& per_core_arg_shapes,
    const std::vector<std::vector<xla::Shape>>& per_core_output_shapes,
    const std::vector<std::vector<std::pair<int, bool>>>&
        per_core_variable_indices,
    const absl::optional<xla::DeviceAssignment>& device_assignment) {
  VLOG(1) << "Compile Tpu programs.";
  std::vector<std::unique_ptr<xla::HloModule>> hlo_modules;
  auto status = CreateHloModules(metadata, compilation_result,
                                 device_assignment, &hlo_modules);
  if (!status.ok()) {
    return status;
  }

  return CompileAheadOfTime(
      absl::make_unique<xla::HloModuleGroup>(hlo_modules[0]->name(),
                                             absl::MakeSpan(hlo_modules)),
      compilation_result, metadata, per_core_arg_shapes, per_core_output_shapes,
      per_core_variable_indices, device_assignment);
}
}  // namespace

void TpuProgramGroup::Initialize(
    absl::Span<XLA_TpuProgram* const> xla_tpu_programs) {
  CHECK_GT(xla_tpu_programs.size(), 0);
  set_tpu_programs(xla_tpu_programs);

  std::vector<bool> may_modify_variables_array(xla_tpu_programs.size(), false);
  std::vector<TPUExecutableInfoProto> executable_infos(xla_tpu_programs.size());
  std::vector<TPUHostTransferInfoProto> host_transfer_infos(
      xla_tpu_programs.size());
  std::vector<xla::HloProto> hlo_metadatas(xla_tpu_programs.size());
  for (size_t i = 0; i < xla_tpu_programs.size(); ++i) {
    const XLA_TpuProgram* xla_tpu_program = xla_tpu_programs[i];
    bool may_modify_variables;
    TpuProgramApiFn()->TpuProgram_GetMayModifyVariablesFn(
        xla_tpu_program, &may_modify_variables);
    may_modify_variables_array[i] = may_modify_variables;

    TpuSerializedProto serialized_executable_info;
    TpuProgramApiFn()->TpuProgram_GetExecutableInfoFn(
        xla_tpu_program, &serialized_executable_info);
    TPUExecutableInfoProto executable_info =
        se_tpu::DeserializeProto<TPUExecutableInfoProto>(
            serialized_executable_info);
    executable_infos[i] = executable_info;
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_executable_info);

    TPUHostTransferInfoProto host_transfer_info;
    TpuSerializedProto serialized_host_transfer_info;
    TpuProgramApiFn()->TpuProgram_GetHostTransferInfoFn(
        xla_tpu_program, &serialized_host_transfer_info);
    if (serialized_host_transfer_info.size > 0) {
      host_transfer_info = se_tpu::DeserializeProto<TPUHostTransferInfoProto>(
          serialized_host_transfer_info);
      StreamExecutor_Tpu_FreeSerializedProto(&serialized_host_transfer_info);
    }
    host_transfer_infos[i] = host_transfer_info;

    TpuSerializedProto serialized_hlo_metadata;
    TpuProgramApiFn()->TpuProgram_GetHloMetadataFn(xla_tpu_program,
                                                   &serialized_hlo_metadata);
    xla::HloProto hlo_metadata =
        se_tpu::DeserializeProto<xla::HloProto>(serialized_hlo_metadata);
    hlo_metadatas[i] = hlo_metadata;
    StreamExecutor_Tpu_FreeSerializedProto(&serialized_hlo_metadata);
  }

  may_modify_variables_ = may_modify_variables_array;
  executable_infos_ = executable_infos;
  host_transfer_infos_ = host_transfer_infos;
  hlo_metadatas_ = hlo_metadatas;
  RefreshHloMetadatasPtrs();
}

bool TpuProgramGroup::has_sharding_program() const {
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    if (!TpuProgramApiFn()->TpuProgram_HasShardingFn(tpu_program)) {
      return false;
    }
  }
  return true;
}

size_t TpuProgramGroup::program_count() const { return tpu_programs_.size(); }

int64_t TpuProgramGroup::program_size() const {
  int64_t total_size = 0;
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    total_size += TpuProgramApiFn()->TpuProgram_GetProgramSizeFn(tpu_program);
  }
  return total_size;
}

bool TpuProgramGroup::LogProgramMemorySummary() {
  bool success = true;
  for (const XLA_TpuProgram* tpu_program : tpu_programs_) {
    success &=
        TpuProgramApiFn()->TpuProgram_LogProgramMemorySummaryFn(tpu_program);
  }
  return success;
}

void TpuProgramGroup::UnloadAndDestroyPrograms() {
  for (XLA_TpuProgram* tpu_program : tpu_programs_) {
    StatusHelper status;
    TpuProgramApiFn()->TpuProgram_UnloadAndDestroyFn(tpu_program,
                                                     status.c_status);
    auto s = status.status();
    if (!s.ok()) {
      LOG(ERROR) << "TpuProgramGroup::UnloadPrograms(): " << s.ToString();
    }
  }
  tpu_programs_.clear();
}

/*static*/ Status TpuProgramGroup::Build(
    const TPUCompileMetadataProto& metadata,
    const tensorflow::XlaCompiler::CompilationResult& compilation_result,
    const std::vector<ShardingAndIndex>& arg_core_mapping,
    const std::vector<std::vector<xla::Shape>>& per_core_arg_shapes,
    const absl::optional<xla::DeviceAssignment>& xla_device_assignment,
    TpuProgramGroupInterface* tpu_program_group_interface) {
  std::vector<std::vector<xla::Shape>> per_core_output_shapes(
      metadata.num_cores_per_replica());
  TF_RETURN_IF_ERROR(ComputeOutputShapesForEachCore(
      metadata, compilation_result, &per_core_output_shapes));

  std::vector<std::vector<std::pair<int, bool>>> per_core_variable_indices(
      metadata.num_cores_per_replica());
  std::vector<bool> may_modify_variables;
  TF_RETURN_IF_ERROR(AddVariableUpdatesToCores(
      metadata, compilation_result, arg_core_mapping, &may_modify_variables,
      &per_core_output_shapes, &per_core_variable_indices));
  TF_RET_CHECK(per_core_arg_shapes.size() == metadata.num_cores_per_replica());
  TF_RET_CHECK(per_core_output_shapes.size() == per_core_arg_shapes.size());
  TF_RET_CHECK(per_core_output_shapes.size() ==
               per_core_variable_indices.size());

  // With shardable input/output pairs, XLA could generate separate
  // sharding/unsharding programs along with the main program. The
  // sharding/unsharding programs will be in nested entries of the AOT
  // compilation result.
  auto status_or = CompileAheadOfTime(
      metadata, compilation_result, per_core_arg_shapes, per_core_output_shapes,
      per_core_variable_indices, xla_device_assignment);

  TF_ASSIGN_OR_RETURN(std::vector<XLA_TpuProgram*> xla_tpu_programs,
                      std::move(status_or));
  // SPMD could return 1 result for all partitions.
  TF_RET_CHECK(xla_tpu_programs.size() == 1 ||
               xla_tpu_programs.size() == metadata.num_cores_per_replica());

  // TODO(henrytan): add an interface to TpuProgramGroupInterface to set
  // may_modify_variables.
  TpuProgramGroup* tpu_program_group =
      tensorflow::down_cast<TpuProgramGroup*>(tpu_program_group_interface);
  tpu_program_group->Initialize(xla_tpu_programs);
  tpu_program_group->may_modify_variables_ = may_modify_variables;
  return Status::OK();
}

TpuProgramGroup::TpuProgramGroup(TpuProgramGroup&& other)
    : may_modify_variables_(std::move(other.may_modify_variables_)),
      tpu_programs_(std::move(other.tpu_programs_)),
      executable_infos_(std::move(other.executable_infos_)),
      host_transfer_infos_(std::move(other.host_transfer_infos_)),
      hlo_metadatas_(std::move(other.hlo_metadatas_)) {
  RefreshHloMetadatasPtrs();
}

void TpuProgramGroup::set_hlo_metadata(const xla::HloProto& hlo_metadata) {
  // TODO(henrytan): initialize hlo_metadatas_ for multi program support.
  if (hlo_metadatas_.empty()) {
    hlo_metadatas_.push_back(hlo_metadata);
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

const std::vector<bool>& TpuProgramGroup::may_modify_variables() const {
  return may_modify_variables_;
}

void TpuProgramGroup::set_may_modify_variables(
    const std::vector<bool>& may_modify_variables) {
  may_modify_variables_ = may_modify_variables;
}

const std::vector<XLA_TpuProgram*>& TpuProgramGroup::tpu_programs() const {
  return tpu_programs_;
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
  CompileApiFn()->TpuCompile_CompileAndBuildFn(serialized_compilation_request,
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
  TpuProgramApiFn()->TpuProgram_FreeArrayFn(xla_tpu_programs);
  return status.status();
}

std::vector<XLA_TpuProgram*> TpuProgramGroup::tpu_programs(
    TpuProgramShardingType sharding_type) const {
  std::vector<XLA_TpuProgram*> tpu_programs;
  tpu_programs.reserve(tpu_programs_.size());
  for (size_t i = 0; i < tpu_programs_.size(); ++i) {
    if (TpuProgramApiFn()->TpuProgram_HasShardingFn(tpu_programs_[i])) {
      tpu_programs.push_back(TpuProgramApiFn()->TpuProgram_GetTpuProgramFn(
          tpu_programs_[i], sharding_type));
      CHECK_NE(tpu_programs[i], nullptr);
    }
  }
  return tpu_programs;
}
}  // namespace tpu
}  // namespace tensorflow
