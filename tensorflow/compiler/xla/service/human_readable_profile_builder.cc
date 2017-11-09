/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/human_readable_profile_builder.h"
#include "tensorflow/compiler/xla/metric_table_report.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {

using tensorflow::strings::Appendf;
using tensorflow::strings::HumanReadableElapsedTime;
using tensorflow::strings::HumanReadableNumBytes;
using tensorflow::strings::StrAppend;

string HumanReadableProfileBuilder::ToString() const {
  string s;

  Appendf(&s, "Execution profile for %s: (%s @ f_nom)\n",
          computation_name_.c_str(),
          HumanReadableElapsedTime(CyclesToSeconds(total_cycles_)).c_str());

  auto append_op = [&](const OpInfo& op) {
    string bytes_per_sec;
    string bytes_per_cycle;
    if (op.cycles <= 0 || op.bytes_accessed < 0) {
      bytes_per_sec = "<unknown>";
      bytes_per_cycle = "<unknown>";
    } else {
      bytes_per_sec =
          HumanReadableNumBytes(op.bytes_accessed / CyclesToSeconds(op.cycles));
      bytes_per_cycle = HumanReadableNumBytes(op.bytes_accessed / op.cycles);
    }

    double cycles_percent = 0;
    if (total_cycles_ > 0) {
      cycles_percent = op.cycles / static_cast<double>(total_cycles_) * 100;
    }

    double nsecs = op.cycles / clock_rate_ghz_;
    Appendf(&s,
            "%15lld cycles (%6.2f%%) :: %12.1f usec (%12.1f optimal) :: %18s "
            ":: %18s :: %12s/s :: %12s/cycle :: %s\n",
            op.cycles, cycles_percent, CyclesToMicroseconds(op.cycles),
            op.optimal_seconds * 1e6,
            op.flop_count <= 0
                ? "<none>"
                : HumanReadableNumFlops(op.flop_count, nsecs).c_str(),
            op.transcendental_count <= 0 ? "<none>"
                                         : HumanReadableNumTranscendentalOps(
                                               op.transcendental_count, nsecs)
                                               .c_str(),
            bytes_per_sec.c_str(), bytes_per_cycle.c_str(), op.name.c_str());
  };

  float optimal_seconds_sum = 0.0;
  int64 total_flops = 0.;
  int64 total_transcendentals = 0.;
  int64 total_bytes = 0;
  for (const auto& op : op_infos_) {
    optimal_seconds_sum += op.optimal_seconds;
    total_flops += op.flop_count;
    total_transcendentals += op.transcendental_count;
    total_bytes += op.bytes_accessed;
  }

  VLOG(1) << "Total floating point ops: " << total_flops;

  append_op({"[total]", "[total]", /*category=*/"", total_cycles_, total_flops,
             total_transcendentals, total_bytes, optimal_seconds_sum});

  // Sort ops in decreasing order of cycles.
  std::vector<OpInfo> sorted_ops(op_infos_);
  std::sort(
      sorted_ops.begin(), sorted_ops.end(),
      [](const OpInfo& a, const OpInfo& b) { return a.cycles > b.cycles; });
  for (const auto& op : sorted_ops) {
    append_op(op);
  }

  if (total_cycles_ <= 0) {
    StrAppend(&s, "****** 0 total cycles ******\n");
  } else {
    // Only show an optimal discrepancy table if at least one value was
    // specified. Estimates are non-negative, so if the sum is greater than
    // zero, then at least one summand was greater than zero.
    if (optimal_seconds_sum > 0) {
      MetricTableReport table;
      table.SetMetricName("microseconds above estimated optimum");
      table.SetEntryName("ops");
      table.SetShowCategoryTable();
      float total_discrepancy_in_microseconds = 0.0f;
      for (const auto& op : sorted_ops) {
        MetricTableReport::Entry entry;
        entry.text = op.name;
        entry.short_text = op.short_name;
        entry.category_text = op.category;
        entry.metric =
            CyclesToMicroseconds(op.cycles) - op.optimal_seconds * 1e6;
        total_discrepancy_in_microseconds += entry.metric;
        table.AddEntry(std::move(entry));
      }
      StrAppend(&s, table.MakeReport(total_discrepancy_in_microseconds));
    }

    {
      MetricTableReport table;
      table.SetMetricName("microseconds");
      table.SetEntryName("ops");
      table.SetShowCategoryTable();
      for (const auto& op : sorted_ops) {
        MetricTableReport::Entry entry;
        entry.text = op.name;
        entry.short_text = op.short_name;
        entry.category_text = op.category;
        entry.metric = CyclesToMicroseconds(op.cycles);
        table.AddEntry(std::move(entry));
      }
      StrAppend(&s, table.MakeReport(CyclesToMicroseconds(total_cycles_)));
    }
  }
  return s;
}

}  // namespace xla
