/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/util/split_proto/split_gpu_executable_writer.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/sort_json.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util/split_proto/split_proto.pb.h"
#include "xla/util/split_proto/split_proto_riegeli_options.h"
#include "xla/util/split_proto/split_proto_write_record.h"
#include "xla/xla.pb.h"

namespace xla {

namespace {

SplitProtoManifest BuildManifest(int32_t num_of_contants) {
  SplitProtoManifest manifest;
  *manifest.mutable_result_proto_type() =
      gpu::GpuExecutableProto::descriptor()->full_name();

  Record::FieldOverrideRecord* override_asm_text =
      manifest.add_records()->mutable_field_override_record();
  override_asm_text->mutable_field_path()->Add()->set_field_number(3);
  override_asm_text->set_field_type(Record::FieldOverrideRecord::TYPE_STRING);

  Record::FieldOverrideRecord* override_binary =
      manifest.add_records()->mutable_field_override_record();
  override_binary->mutable_field_path()->Add()->set_field_number(4);
  override_binary->set_field_type(Record::FieldOverrideRecord::TYPE_STRING);

  // dnn_compiled_graphs
  manifest.add_records()->mutable_proto_merge_record();
  // constants may be quite big, so put each of them in a separate record
  for (int i = 0; i < num_of_contants; ++i) {
    manifest.add_records()->mutable_proto_merge_record();
  }
  // The rest of the fields (i.e. the non-offloaded fields)
  manifest.add_records()->mutable_proto_merge_record();

  return manifest;
}

void NormalizeLiteralProto(LiteralProto* literal) {
  if (!literal) return;
  for (int i = 0; i < literal->f32s_size(); ++i) {
    if (std::isnan(literal->f32s(i))) {
      literal->set_f32s(i, std::numeric_limits<float>::quiet_NaN());
    }
  }
  for (int i = 0; i < literal->f64s_size(); ++i) {
    if (std::isnan(literal->f64s(i))) {
      literal->set_f64s(i, std::numeric_limits<double>::quiet_NaN());
    }
  }
  for (int i = 0; i < literal->c64s_size(); ++i) {
    if (std::isnan(literal->c64s(i))) {
      literal->set_c64s(i, std::numeric_limits<float>::quiet_NaN());
    }
  }
  for (int i = 0; i < literal->c128s_size(); ++i) {
    if (std::isnan(literal->c128s(i))) {
      literal->set_c128s(i, std::numeric_limits<double>::quiet_NaN());
    }
  }
  for (int i = 0; i < literal->tuple_literals_size(); ++i) {
    NormalizeLiteralProto(literal->mutable_tuple_literals(i));
  }
}

// If the backend config is a json string, sort the keys to ensure that the
// serialized form is consistent. This is needed to make the output
// deterministic because the order of keys in json is not guaranteed.
//
// If the backend config is not a json string, it is not modified.
absl::Status NormalizeBackendConfig(gpu::GpuExecutableProto& executable) {
  for (HloComputationProto& computation :
       *executable.mutable_hlo_module_with_config()
            ->mutable_hlo_module()
            ->mutable_computations()) {
    for (HloInstructionProto& instruction :
         *computation.mutable_instructions()) {
      if (instruction.has_literal()) {
        NormalizeLiteralProto(instruction.mutable_literal());
      }
      auto backend_config_str_or = GetBackendConfigString(
          instruction, &executable.hlo_module_with_config().hlo_module());
      if (!backend_config_str_or.ok()) {
        return backend_config_str_or.status();
      }
      std::string backend_config_str = std::move(backend_config_str_or).value();
      auto normalized_or = SortJson(backend_config_str);
      if (!normalized_or.ok()) {
        continue;
      }
      std::string normalized = std::move(normalized_or).value();
      if (normalized == backend_config_str) {
        continue;
      }
      if (instruction.has_backend_config_payload()) {
        Payload* payload = instruction.mutable_backend_config_payload();
        switch (payload->payload_source_case()) {
          case Payload::kId: {
            int id = static_cast<int>(payload->id());
            auto* module = executable.mutable_hlo_module_with_config()
                               ->mutable_hlo_module();
            *module->mutable_payloads(id) = normalized;
            break;
          }
          case Payload::kValue:
          case Payload::PAYLOAD_SOURCE_NOT_SET:
            payload->set_value(normalized);
            break;
        }
      } else {
        instruction.set_backend_config(normalized);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status WriteSplitGpuExecutable(gpu::GpuExecutableProto executable,
                                     std::unique_ptr<riegeli::Writer> writer) {
  riegeli::RecordWriter record_writer(std::move(writer),
                                      GetSplitProtoRiegeliOptions());
  SplitProtoManifest manifest = BuildManifest(executable.constants_size());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      WriteRecord(record_writer, manifest),
      "failed to write manifest in GpuExecutableProto split proto");

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      WriteRecord(record_writer, executable.asm_text()),
      "failed to serialize asm_text in GpuExecutableProto split proto");
  executable.clear_asm_text();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      WriteRecord(record_writer, executable.binary()),
      "failed to serialize binary in GpuExecutableProto split proto");
  executable.clear_binary();

  gpu::GpuExecutableProto dnn_graphs_wrapper;
  *dnn_graphs_wrapper.mutable_dnn_compiled_graphs() =
      std::move(executable.dnn_compiled_graphs());
  executable.clear_dnn_compiled_graphs();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      WriteRecord(record_writer, dnn_graphs_wrapper),
      "failed to serialize dnn_compiled_graphs in GpuExecutableProto split "
      "proto");

  for (gpu::GpuExecutableProto::ConstantInfoProto& constant :
       *executable.mutable_constants()) {
    gpu::GpuExecutableProto constant_wrapper;
    *constant_wrapper.add_constants() = std::move(constant);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        WriteRecord(record_writer, constant_wrapper),
        "failed to serialize constant in GpuExecutableProto split proto");
  }
  executable.clear_constants();

  // The rest of the fields (i.e. the non-offloaded fields)
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      NormalizeBackendConfig(executable),
      "failed to normalize backend config in GpuExecutableProto split proto");
  // Module IDs are created via a static counter when deserializing, and they
  // can cause non-determinism, so we don't preserve them.
  executable.mutable_hlo_module_with_config()->mutable_hlo_module()->clear_id();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      WriteRecord(record_writer, executable),
      "failed to serialize the rest of the fields in GpuExecutableProto "
      "split proto (all fields except asm_text, binary, "
      "dnn_compiled_graphs, and constants)");

  if (!record_writer.Close()) {
    return record_writer.status();
  }
  return absl::OkStatus();
}

}  // namespace xla
