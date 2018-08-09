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
using tensorflow::strings::Printf;
using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;

string HumanReadableProfileBuilder::ToString() const {
  string s;

  Appendf(&s, "Execution profile for %s: (%s @ f_nom)\n",
          computation_name_.c_str(),
          HumanReadableElapsedTime(CyclesToSeconds(total_cycles_)).c_str());

  int64 cumulative_cycles = 0;
  auto print_op = [&](const OpInfo& op, bool is_total = false) {
    // Skip ops with 0 optimal seconds and 0 actual cycles.  These are ops that
    // were expected to be free and are actually free -- things like (on most
    // backends) kParameter or kConstant HLOs.  There's no need to clutter the
    // profile with these.
    if (op.optimal_seconds == 0 && op.cycles == 0) {
      return;
    }

    string bytes_per_sec;
    string bytes_per_cycle;
    if (op.cycles > 0 && op.bytes_accessed >= 0) {
      bytes_per_sec = StrCat(
          HumanReadableNumBytes(op.bytes_accessed / CyclesToSeconds(op.cycles)),
          "/s");
      double bpc = static_cast<double>(op.bytes_accessed) / op.cycles;
      if (op.bytes_accessed > op.cycles) {
        bytes_per_cycle = StrCat(HumanReadableNumBytes(bpc), "/cycle");
      } else {
        bytes_per_cycle = Printf("%.3fB/cycle", bpc);
      }
    }

    double cumulative_cycles_percent = 0;
    double cycles_percent = 0;
    if (!is_total) {
      cumulative_cycles += op.cycles;
    }
    if (total_cycles_ > 0) {
      cycles_percent = op.cycles / static_cast<double>(total_cycles_) * 100;
      cumulative_cycles_percent =
          cumulative_cycles / static_cast<double>(total_cycles_) * 100;
    }

    string cycles_percent_str;
    if (is_total) {
      // Leaving off the two trailing decimal points of "100.%" lets us save two
      // columns in the output.
      cycles_percent_str = "100.% 100Σ";
    } else {
      cycles_percent_str =
          Printf("%5.2f%% %2.0fΣ", cycles_percent, cumulative_cycles_percent);
    }

    double nsecs = op.cycles / clock_rate_ghz_;
    Appendf(
        &s,
        "%15lld cycles (%s) :: %12.1f usec %22s :: %18s :: %18s :: %14s :: "
        "%16s :: %s\n",
        op.cycles, cycles_percent_str.c_str(), CyclesToMicroseconds(op.cycles),
        op.optimal_seconds < 0
            ? ""
            : Printf("(%12.1f optimal)", op.optimal_seconds * 1e6).c_str(),
        op.flop_count <= 0
            ? ""
            : HumanReadableNumFlops(op.flop_count, nsecs).c_str(),
        op.transcendental_count <= 0
            ? ""
            : HumanReadableNumTranscendentalOps(op.transcendental_count, nsecs)
                  .c_str(),
        bytes_per_sec.c_str(), bytes_per_cycle.c_str(), op.name.c_str());
  };

  float optimal_seconds_sum = 0.0;
  int64 total_flops = 0.;
  int64 total_transcendentals = 0.;
  int64 total_bytes = 0;
  for (const auto& op : op_infos_) {
    if (op.optimal_seconds > 0) {
      optimal_seconds_sum += op.optimal_seconds;
    }
    total_flops += std::max(op.flop_count, int64{0});
    total_transcendentals += std::max(op.transcendental_count, int64{0});
    total_bytes += std::max(op.bytes_accessed, int64{0});
  }

  VLOG(1) << "Total floating point ops: " << total_flops;

  print_op({"[total]", "[total]", /*category=*/"", total_cycles_, total_flops,
            total_transcendentals, total_bytes, optimal_seconds_sum},
           /*is_total=*/true);

  // Sort ops in decreasing order of cycles, and print them.
  std::vector<OpInfo> sorted_ops(op_infos_);
  std::sort(
      sorted_ops.begin(), sorted_ops.end(),
      [](const OpInfo& a, const OpInfo& b) { return a.cycles > b.cycles; });
  for (const auto& op : sorted_ops) {
    print_op(op);
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
      table.SetShowAllEntries();
      float total_discrepancy_in_microseconds = 0.0f;
      for (const auto& op : op_infos_) {
        // Skip ops with < 0 optimal seconds.  These are ops for which we don't
        // know the optimal time.
        if (op.optimal_seconds < 0) {
          continue;
        }
        // Also skip ops with 0 actual cycles.  These ops were free; there's no
        // need to clutter the "above estimated optimum" table with them,
        // because they can't be optimized further.
        if (op.cycles == 0) {
          continue;
        }
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
      table.SetShowAllEntries();
      for (const auto& op : op_infos_) {
        // Skip ops with 0 optimal seconds and 0 actual cycles.  As in
        // print_op(), these are uninteresting because they're expected to be
        // free, and they were actually free.
        if (op.cycles == 0 && op.optimal_seconds == 0) {
          continue;
        }
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

  if (total_bytes > 0) {
    MetricTableReport table;
    table.SetMetricName("MiB read+written");
    table.SetEntryName("ops");
    table.SetShowCategoryTable();
    for (const auto& op : op_infos_) {
      MetricTableReport::Entry entry;
      entry.text = op.name;
      entry.short_text = op.short_name;
      entry.category_text = op.category;
      entry.metric = static_cast<double>(op.bytes_accessed) / (1 << 20);
      table.AddEntry(std::move(entry));
    }
    StrAppend(&s,
              table.MakeReport(static_cast<double>(total_bytes) / (1 << 20)));
  }
  return s;
}

}  // namespace xla
