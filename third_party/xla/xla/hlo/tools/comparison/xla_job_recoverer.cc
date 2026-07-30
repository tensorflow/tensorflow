/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/xla_job_recoverer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_calculator.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_propagator.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_sequencer.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {

using DeviceTensorSummary = XlaJobRecoverer::DeviceTensorSummary;
using LogicalID = xla::DeviceAssignment::LogicalID;
using ShardTensorSummary = OriginalTensorSummaryCalculator::ShardTensorSummary;

struct XlaJobRecovererData {
  struct PerReplicaWriter {
    std::unique_ptr<riegeli::RecordWriterBase> raw_summary_writer;
    std::string raw_summary_path;
  };
  std::vector<PerReplicaWriter> per_replica_writers;
  std::vector<std::unique_ptr<OriginalTensorSummaryCalculator>> calculators;
  std::unique_ptr<const xla::DeviceAssignment> device_assignment;
  HloModule* original_module;
  XlaJobRecoverer::OriginalTensorSummaryCallbackGetter callback_getter;
  std::string temp_file_base_path;
  std::string sequenced_file_base_path;
  ComparisonVariant comparison_variant;
};

namespace {

absl::StatusOr<std::unique_ptr<riegeli::RecordWriterBase>> CreateRecordWriter(
    absl::string_view path) {
  std::unique_ptr<riegeli::RecordWriterBase> writer =
      riegeli::Maker<riegeli::RecordWriter>(
          riegeli::Maker<riegeli::FdWriter>(path),
          riegeli::RecordWriterBase::Options().set_transpose(true));
  if (!writer->ok()) {
    return writer->status();
  }
  return writer;
}

}  // namespace

XlaJobRecoverer::XlaJobRecoverer(std::unique_ptr<XlaJobRecovererData> data)
    : data_(std::move(data)) {}

XlaJobRecoverer::~XlaJobRecoverer() = default;

/*static*/ absl::StatusOr<
    std::pair<std::unique_ptr<XlaJobRecoverer>,
              OriginalTensorSummaryCalculator::CreationMetrics>>
XlaJobRecoverer::Create(
    std::unique_ptr<const xla::DeviceAssignment> device_assignment,
    HloModule* original_module, HloModule* optimized_module,
    OriginalTensorSummaryCallbackGetter&& callback_getter,
    absl::string_view temp_file_base_path,
    absl::string_view sequenced_file_base_path,
    ComparisonVariant comparison_variant) {
  if (device_assignment->replica_count() <= 0) {
    return absl::InvalidArgumentError(
        "Device assignment has non-positive replica count.");
  }

  auto data = std::make_unique<XlaJobRecovererData>();
  data->device_assignment = std::move(device_assignment);
  data->original_module = original_module;
  data->callback_getter = std::move(callback_getter);
  data->temp_file_base_path = temp_file_base_path;
  data->sequenced_file_base_path = sequenced_file_base_path;
  data->comparison_variant = comparison_variant;

  data->per_replica_writers.reserve(data->device_assignment->replica_count());
  data->calculators.reserve(data->device_assignment->replica_count());

  // Create the directory for temp files if it doesn't exist.
  RETURN_IF_ERROR(tsl::Env::Default()->RecursivelyCreateDir(
      std::string(tsl::io::Dirname(temp_file_base_path))));
  std::string variant_name = ToString(comparison_variant);
  const std::string raw_path_0 = absl::StrCat(
      temp_file_base_path, ".", variant_name, ".raw_replica_0.riegeli");
  ASSIGN_OR_RETURN(auto writer_0, CreateRecordWriter(raw_path_0));
  data->per_replica_writers.push_back(
      {/*raw_summary_writer=*/std::move(writer_0),
       /*raw_summary_path=*/raw_path_0});

  std::pair<std::unique_ptr<OriginalTensorSummaryCalculator>,
            OriginalTensorSummaryCalculator::CreationMetrics>
      calculator_with_metrics;
  ASSIGN_OR_RETURN(
      calculator_with_metrics,
      OriginalTensorSummaryCalculator::Create(
          optimized_module, original_module,
          [writer = data->per_replica_writers[0].raw_summary_writer.get()](
              const AbsoluteScopedTensorKey& original_tensor_key,
              std::shared_ptr<const tensor_transformation::TensorTransformation>
                  pending_transformation,
              const OriginalTensorSummary& original_tensor_summary) {
            if (!writer->WriteRecord(CreateRecoveredTensorSummaryProto(
                    original_tensor_key, pending_transformation,
                    original_tensor_summary))) {
              return writer->status();
            }
            return absl::OkStatus();
          }));
  data->calculators.push_back(std::move(calculator_with_metrics.first));

  for (int64_t i = 1; i < data->device_assignment->replica_count(); ++i) {
    const std::string raw_path = absl::StrCat(
        temp_file_base_path, ".", variant_name, ".raw_replica_", i, ".riegeli");
    ASSIGN_OR_RETURN(auto writer, CreateRecordWriter(raw_path));
    data->per_replica_writers.push_back(
        {/*raw_summary_writer=*/std::move(writer),
         /*raw_summary_path=*/raw_path});
    data->calculators.push_back(data->calculators[0]->CloneWithCallback(
        [writer = data->per_replica_writers[i].raw_summary_writer.get()](
            const AbsoluteScopedTensorKey& original_tensor_key,
            std::shared_ptr<const tensor_transformation::TensorTransformation>
                pending_transformation,
            const OriginalTensorSummary& original_tensor_summary) {
          if (!writer->WriteRecord(CreateRecoveredTensorSummaryProto(
                  original_tensor_key, pending_transformation,
                  original_tensor_summary))) {
            return writer->status();
          }
          return absl::OkStatus();
        }));
  }

  return std::make_pair(absl::WrapUnique(new XlaJobRecoverer(std::move(data))),
                        calculator_with_metrics.second);
}

absl::Status XlaJobRecoverer::ProcessDeviceTensorSummary(
    const AbsoluteScopedTensorKey& optimized_tensor_position,
    DeviceTensorSummary shard_summary) {
  ASSIGN_OR_RETURN(LogicalID logical_id,
                   data_->device_assignment->LogicalIdForDevice(
                       shard_summary.logical_device_id));
  CHECK_LT(logical_id.replica_id, data_->calculators.size());
  OriginalTensorSummaryCalculator& calculator =
      *data_->calculators[logical_id.replica_id];
  return calculator.ProcessShardSummary(
      optimized_tensor_position,
      ShardTensorSummary{
          /*logical_shard_id=*/logical_id.computation_id,
          /*summary=*/shard_summary.summary,
      });
}

absl::StatusOr<std::vector<OriginalTensorSummaryPropagator::ProcessingMetrics>>
XlaJobRecoverer::Finish() {
  std::vector<OriginalTensorSummaryPropagator::ProcessingMetrics> metrics;
  std::vector<std::string> propagated_paths;
  propagated_paths.reserve(data_->per_replica_writers.size());
  for (int i = 0; i < data_->per_replica_writers.size(); ++i) {
    auto& writer = data_->per_replica_writers[i];
    if (!writer.raw_summary_writer->Close()) {
      return writer.raw_summary_writer->status();
    }

    ASSIGN_OR_RETURN(auto sequencer, OriginalTensorSummarySequencer::Create(
                                         data_->original_module));
    std::string variant_name = ToString(data_->comparison_variant);
    const std::string sequenced_path =
        absl::StrCat(data_->sequenced_file_base_path, ".", variant_name,
                     ".sequenced_replica_", i, ".riegeli");
    ASSIGN_OR_RETURN(
        auto is_recovered_cb,
        sequencer->Sequence(writer.raw_summary_path, sequenced_path));

    const std::string propagated_path =
        absl::StrCat(data_->temp_file_base_path, ".", variant_name,
                     ".propagated_replica_", i, ".riegeli");
    ASSIGN_OR_RETURN(auto propagated_writer,
                     CreateRecordWriter(propagated_path));
    OriginalTensorSummaryPropagator propagator(
        data_->original_module,
        [writer = propagated_writer.get()](
            const AbsoluteScopedTensorKey& original_tensor_key,
            std::shared_ptr<const tensor_transformation::TensorTransformation>
                pending_transformation,
            const OriginalTensorSummary& original_tensor_summary) {
          if (!writer->WriteRecord(CreateRecoveredTensorSummaryProto(
                  original_tensor_key, pending_transformation,
                  original_tensor_summary))) {
            return writer->status();
          }
          return absl::OkStatus();
        },
        std::move(*is_recovered_cb));
    RETURN_IF_ERROR(propagator.Initialize());

    riegeli::RecordReader sequenced_reader(
        riegeli::Maker<riegeli::FdReader>(sequenced_path));
    RecoveredTensorSummaryProto summary_proto;
    while (sequenced_reader.ReadRecord(summary_proto)) {
      ASSIGN_OR_RETURN(auto summary,
                       RecoveredTensorSummaryFromProto(summary_proto));
      RETURN_IF_ERROR(propagator.Process(summary.original_tensor_key,
                                         summary.pending_transformation,
                                         summary.original_tensor_summary));
    }
    RETURN_IF_ERROR(propagator.Finish());
    metrics.push_back(propagator.GetProcessingMetrics());
    if (!sequenced_reader.Close()) {
      return sequenced_reader.status();
    }
    if (!propagated_writer->Close()) {
      return propagated_writer->status();
    }
    propagated_paths.push_back(propagated_path);
  }

  for (int i = 0; i < propagated_paths.size(); ++i) {
    const auto& propagated_path = propagated_paths[i];
    riegeli::RecordReader propagated_reader(
        riegeli::Maker<riegeli::FdReader>(propagated_path));
    RecoveredTensorSummaryProto summary_proto;
    OriginalTensorSummaryCallback callback = data_->callback_getter(i);
    while (propagated_reader.ReadRecord(summary_proto)) {
      ASSIGN_OR_RETURN(auto summary,
                       RecoveredTensorSummaryFromProto(summary_proto));
      RETURN_IF_ERROR(callback(summary.original_tensor_key,
                               summary.pending_transformation,
                               summary.original_tensor_summary));
    }
    if (!propagated_reader.Close()) {
      return propagated_reader.status();
    }
  }

  return metrics;
}

}  // namespace xla::numerics::comparison
