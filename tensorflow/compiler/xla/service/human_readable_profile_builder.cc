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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/metric_table_report.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace xla {

using absl::StrAppend;
using absl::StrAppendFormat;
using absl::StrCat;
using absl::StrFormat;
using tensorflow::strings::HumanReadableElapsedTime;
using tensorflow::strings::HumanReadableNumBytes;

string HumanReadableProfileBuilder::ToString() const {
  string s;

  StrAppendFormat(&s, "Execution profile for %s: (%s @ f_nom)\n",
                  computation_name_,
                  HumanReadableElapsedTime(CyclesToSeconds(total_cycles_)));

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
        bytes_per_cycle = StrFormat("%.3fB/cycle", bpc);
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
      cycles_percent_str = StrFormat("%5.2f%% %2.0fΣ", cycles_percent,
                                     cumulative_cycles_percent);
    }

    double nsecs = op.cycles / clock_rate_ghz_;
    StrAppendFormat(
        &s,
        "%15d cycles (%s) :: %12.1f usec %22s :: %18s :: %18s :: %14s :: "
        "%16s :: %s\n",
        op.cycles, cycles_percent_str, CyclesToMicroseconds(op.cycles),
        op.optimal_seconds < 0
            ? ""
            : StrFormat("(%12.1f optimal)", op.optimal_seconds * 1e6),
        op.flop_count > 0 && nsecs > 0
            ? HumanReadableNumFlops(op.flop_count, nsecs)
            : "",
        op.transcendental_count > 0 && nsecs > 0
            ? HumanReadableNumTranscendentalOps(op.transcendental_count, nsecs)
            : "",
        bytes_per_sec, bytes_per_cycle, op.name);
  };

  double optimal_seconds_sum = 0;
  int64 total_flops = 0.;
  int64 total_transcendentals = 0.;
  int64 total_bytes = 0;
  for (const auto& op : op_infos_) {
    if (op.optimal_seconds > 0) {
      // An op can run faster than the estimated optimum. For example, we might
      // estimate a fusion's speed by looking at the size of its operands and
      // result, but perhaps the fusion doesn't read the entirety of all of its
      // inputs.  For the purposes of summing the instructions' optimal speeds,
      // we treat the "optimum" as the smallest of either the estimated optimum
      // and the actual speed.
      optimal_seconds_sum +=
          std::min(double{op.optimal_seconds}, CyclesToSeconds(op.cycles));
    }
    total_flops += std::max(op.flop_count, int64{0});
    total_transcendentals += std::max(op.transcendental_count, int64{0});
    total_bytes += std::max(op.bytes_accessed, int64{0});
  }

  VLOG(1) << "Total floating point ops: " << total_flops;

  print_op({is_entry_computation_ ? "[total] [entry]" : "[total]", "[total]",
            /*category=*/"", total_cycles_, total_flops, total_transcendentals,
            total_bytes, static_cast<float>(optimal_seconds_sum)},
           /*is_total=*/true);

  // Sort ops in decreasing order of cycles, and print them.
  std::vector<OpInfo> sorted_ops(op_infos_);
  absl::c_sort(sorted_ops, [](const OpInfo& a, const OpInfo& b) {
    return a.cycles > b.cycles;
  });
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
        // Ignore ops that run faster than the estimated optimal here, as we do
        // when calculating optimal_seconds_sum.
        entry.metric = std::max(
            0., CyclesToMicroseconds(op.cycles) - op.optimal_seconds * 1e6);
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
