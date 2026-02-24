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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util/split_proto/split_proto.pb.h"
#include "xla/util/split_proto/split_proto_riegeli_options.h"

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

template <typename T, typename Src>
absl::Status WriteRecord(riegeli::RecordWriter<Src>& record_writer, T& record) {
  if (!record_writer.WriteRecord(record)) {
    return record_writer.status().ok()
               ? absl::InternalError("Failed to write record")
               : record_writer.status();
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status WriteSplitGpuExecutable(gpu::GpuExecutableProto executable,
                                     std::unique_ptr<riegeli::Writer> writer) {
  riegeli::RecordWriter record_writer(std::move(writer),
                                      GetSplitProtoRiegeliOptions());
  SplitProtoManifest manifest = BuildManifest(executable.constants_size());
  TF_RETURN_IF_ERROR(WriteRecord(record_writer, manifest));

  TF_RETURN_IF_ERROR(WriteRecord(record_writer, executable.asm_text()));
  executable.clear_asm_text();
  TF_RETURN_IF_ERROR(WriteRecord(record_writer, executable.binary()));
  executable.clear_binary();

  gpu::GpuExecutableProto dnn_graphs_wrapper;
  *dnn_graphs_wrapper.mutable_dnn_compiled_graphs() =
      std::move(executable.dnn_compiled_graphs());
  executable.clear_dnn_compiled_graphs();
  TF_RETURN_IF_ERROR(WriteRecord(record_writer, dnn_graphs_wrapper));

  for (gpu::GpuExecutableProto::ConstantInfoProto& constant :
       *executable.mutable_constants()) {
    gpu::GpuExecutableProto constant_wrapper;
    *constant_wrapper.add_constants() = std::move(constant);
    TF_RETURN_IF_ERROR(WriteRecord(record_writer, constant_wrapper));
  }
  executable.clear_constants();

  // The rest of the fields (i.e. the non-offloaded fields)
  TF_RETURN_IF_ERROR(WriteRecord(record_writer, executable));

  if (!record_writer.Close()) {
    return record_writer.status();
  }
  return absl::OkStatus();
}

}  // namespace xla
