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
#include "tensorflow/core/profiler/convert/dcn_slack_analysis_combiner.h"

#include <cstdint>

#include "xla/tsl/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/protobuf/dcn_slack_analysis.pb.h"

namespace tensorflow {
namespace profiler {

using tensorflow::profiler::DcnSlackAnalysis;
using tensorflow::profiler::DcnSlackSummary;
using tsl::profiler::SafeDivide;

void DcnSlackAnalysisCombiner::Combine(const DcnSlackAnalysis& slack_analysis) {
  for (const auto& slack : slack_analysis.dcn_slack_summary()) {
    uint64_t occurrences = slack.occurrences();
    DcnSlackSummary& summary = slack_summary_[slack.rendezvous()];
    summary.set_slack_us(summary.slack_us() + slack.slack_us() * occurrences);
    summary.set_observed_duration_us(summary.observed_duration_us() +
                                     slack.observed_duration_us() *
                                         occurrences);
    summary.set_stall_duration_us(summary.stall_duration_us() +
                                  slack.stall_duration_us() * occurrences);
    summary.set_send_done_duration_us(summary.send_done_duration_us() +
                                      slack.send_done_duration_us() *
                                          occurrences);
    summary.set_recv_done_duration_us(summary.recv_done_duration_us() +
                                      slack.recv_done_duration_us() *
                                          occurrences);
    summary.set_send_duration_us(summary.send_duration_us() +
                                 slack.send_duration_us() * occurrences);
    summary.set_recv_duration_us(summary.recv_duration_us() +
                                 slack.recv_duration_us() * occurrences);
    summary.set_host_stall_us(summary.host_stall_us() +
                              slack.host_stall_us() * occurrences);
    summary.set_occurrences(summary.occurrences() + slack.occurrences());
    summary.set_bytes_transmitted_over_network(
        slack.bytes_transmitted_over_network());
    summary.set_recv_op_name(slack.recv_op_name());
    summary.set_send_op_name(slack.send_op_name());
    summary.set_transfer_type(slack.transfer_type());
  }
}

DcnSlackAnalysis DcnSlackAnalysisCombiner::Finalize() {
  DcnSlackAnalysis analysis;
  for (const auto& [rendezvous, summary] : slack_summary_) {
    auto* slack = analysis.add_dcn_slack_summary();
    slack->set_rendezvous(rendezvous);
    slack->set_recv_op_name(summary.recv_op_name());
    slack->set_send_op_name(summary.send_op_name());
    slack->set_transfer_type(summary.transfer_type());
    slack->set_slack_us(SafeDivide(summary.slack_us(), summary.occurrences()));
    slack->set_observed_duration_us(
        SafeDivide(summary.observed_duration_us(), summary.occurrences()));
    slack->set_stall_duration_us(
        SafeDivide(summary.stall_duration_us(), summary.occurrences()));
    slack->set_send_done_duration_us(
        SafeDivide(summary.send_done_duration_us(), summary.occurrences()));
    slack->set_recv_done_duration_us(
        SafeDivide(summary.recv_done_duration_us(), summary.occurrences()));
    slack->set_send_duration_us(
        SafeDivide(summary.send_duration_us(), summary.occurrences()));
    slack->set_recv_duration_us(
        SafeDivide(summary.recv_duration_us(), summary.occurrences()));
    slack->set_host_stall_us(
        SafeDivide(summary.host_stall_us(), summary.occurrences()));
    slack->set_occurrences(summary.occurrences());
    slack->set_bytes_transmitted_over_network(
        summary.bytes_transmitted_over_network());
  }

  return analysis;
}

}  // namespace profiler
}  // namespace tensorflow
