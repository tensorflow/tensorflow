/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_XSPACE_TO_DCN_SLACK_ANALYSIS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_XSPACE_TO_DCN_SLACK_ANALYSIS_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"
#include "tensorflow/core/profiler/utils/hlo_proto_map.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

using tensorflow::profiler::DcnSlackAnalysis;

namespace dcn_analysis_internal {

struct DcnOpState {
  uint64_t start_time = 0;
  uint64_t end_time = 0;

  // Duration of containing send/send-done/recv/recv-done ops that needs to be
  // subtracted from the total duration
  uint64_t overlapping_duration = 0;
  std::string rendezvous_name;
  uint64_t stall_duration_ns = 0;
  std::string send_op_name;
};

struct InstrMetadata {
  xla::HloOpcode opcode;
  uint64_t channel_id;
  std::optional<std::string> rendezvous_name;
  int64_t size = 0;
};

class DcnTracker {
 public:
  explicit DcnTracker(const tensorflow::profiler::HloProtoMap& hlo_proto_map)
      : hlo_proto_map_(hlo_proto_map) {}

  absl::StatusOr<InstrMetadata> GetInstructionMetadata(std::string_view module,
                                                       std::string_view instr);

  DcnSlackAnalysis Finalize();

  void DebugString();

  void VisitOp(const InstrMetadata& instr,
               const tsl::profiler::XEventVisitor& visitor);

 private:
  DcnSlackAnalysis slack_analysis_;
  absl::flat_hash_map<std::string, DcnOpState> rendezvous_to_op_map_;
  absl::flat_hash_map<uint64_t, std::string> channel_id_to_rendezvous_map_;
  absl::flat_hash_map<std::string, InstrMetadata> instruction_metadata_map_;
  const tensorflow::profiler::HloProtoMap& hlo_proto_map_;
  absl::flat_hash_map<std::string, std::unique_ptr<xla::HloModule>>
      hlo_module_cache_;

  absl::StatusOr<InstrMetadata> GetInstrMetadataFromHloModule(
      std::string_view module, std::string_view instr);

  void UpdateActiveOps(uint64_t duration);

  void SummarizeDcnSlackAnalysis();
};

}  // namespace dcn_analysis_internal

// Convert Hlo Events in XSpace to Dcn Slack analysis.
DcnSlackAnalysis ConvertXSpaceToDcnSlackAnalysis(
    const tensorflow::profiler::XSpace& xspace);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_XSPACE_TO_DCN_SLACK_ANALYSIS_H_
