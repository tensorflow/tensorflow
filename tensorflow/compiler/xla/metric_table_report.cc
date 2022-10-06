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

#include "tensorflow/compiler/xla/metric_table_report.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

void MetricTableReport::AddEntry(Entry entry) {
  entries_.push_back(std::move(entry));
}

void MetricTableReport::SetMetricName(std::string metric_name) {
  metric_name_ = std::move(metric_name);
}

void MetricTableReport::SetEntryName(std::string entry_name) {
  entry_name_ = std::move(entry_name);
}

void MetricTableReport::SetShowAllEntries() {
  max_entries_to_show_ = std::numeric_limits<int64_t>::max();
  max_entries_per_category_to_show_ = std::numeric_limits<int64_t>::max();
  max_metric_proportion_to_show_ = 1.1;  // more than 100%
}

void MetricTableReport::SetShowCategoryTable() { show_category_table_ = true; }

void MetricTableReport::SetShowEntryTable() { show_entry_table_ = true; }

std::string MetricTableReport::MakeReport(double expected_metric_sum) {
  expected_metric_sum_ = expected_metric_sum;
  report_.clear();

  // Sort the entries.
  const auto metric_greater = [](const Entry& a, const Entry& b) {
    return a.metric > b.metric;
  };
  absl::c_sort(entries_, metric_greater);

  // Create the report
  AppendLine();
  AppendHeader();

  if (show_category_table_) {
    AppendLine();
    AppendCategoryTable();
  }
  if (show_entry_table_) {
    AppendLine();
    AppendEntryTable();
  }
  AppendLine();

  return std::move(report_);
}

void MetricTableReport::WriteReportToInfoLog(double expected_metric_sum) {
  // Write something to the log normally to get the date-time and file prefix.
  LOG(INFO) << "Writing report to log.";

  int64_t pos = 0;
  const std::string report = MakeReport(expected_metric_sum);
  const int report_size = report.size();
  while (pos < report_size) {
    int64_t end_of_line = report.find('\n', pos);
    const int64_t _npos = std::string::npos;
    if (end_of_line == _npos) {
      end_of_line = report.size();
    }
    absl::string_view line(report.data() + pos, end_of_line - pos);

    // TODO(b/34779244): Figure out how to do this without the verbose log-line
    // prefix. The usual way didn't compile on open source.
    LOG(INFO) << line;

    pos = end_of_line + 1;
  }
}

std::vector<MetricTableReport::Category> MetricTableReport::MakeCategories(
    const std::vector<Entry>* entries) {
  // Create the categories using a category_text -> category map.
  absl::flat_hash_map<std::string, Category> category_map;
  for (const Entry& entry : *entries) {
    Category& category = category_map[entry.category_text];
    category.metric_sum += entry.metric;
    category.entries.push_back(&entry);
  }

  // Move the categories to a vector.
  std::vector<Category> categories;
  categories.reserve(category_map.size());
  for (auto& key_value_pair : category_map) {
    categories.push_back(std::move(key_value_pair.second));
    categories.back().category_text = key_value_pair.first;
  }

  // Sort the categories.
  auto metric_sum_greater = [](const Category& a, const Category& b) {
    return a.metric_sum > b.metric_sum;
  };
  absl::c_sort(categories, metric_sum_greater);

  return categories;
}

void MetricTableReport::AppendHeader() {
  AppendLine("********** ", metric_name_, " report **********");
  AppendLine("There are ", MetricString(expected_metric_sum_), " ",
             metric_name_, " in total.");
  AppendLine("There are ", MetricString(UnaccountedMetric()), " ", metric_name_,
             " (", MetricPercent(UnaccountedMetric()),
             ") not accounted for by the data.");
  AppendLine("There are ", entries_.size(), " ", entry_name_, ".");
}

void MetricTableReport::AppendCategoryTable() {
  const std::vector<Category> categories = MakeCategories(&entries_);

  AppendLine("********** categories table for ", metric_name_, " **********");
  AppendLine();

  double metric_sum = UnaccountedMetric();
  int64_t categories_shown = 0;
  for (const auto& category : categories) {
    if (categories_shown >= max_entries_to_show_ ||
        metric_sum / expected_metric_sum_ > max_metric_proportion_to_show_) {
      break;
    }
    ++categories_shown;
    metric_sum += category.metric_sum;

    // Show the category.
    std::string text = category.category_text;
    if (text.empty()) {
      text = "[no category]";
    }
    absl::StrAppend(&text, " (", category.entries.size(), " ", entry_name_,
                    ")");
    AppendTableRow(text, category.metric_sum, metric_sum);

    // Show the top entries in the category.
    const char* const kIndentPrefix = "                              * ";
    int64_t entries_to_show = std::min<int64_t>(
        max_entries_per_category_to_show_, category.entries.size());
    const int64_t category_entries_size = category.entries.size();
    if (category_entries_size == entries_to_show + 1) {
      // May as well show the last entry on the line that would otherwise say
      // that there is a single entry not shown.
      ++entries_to_show;
    }
    for (int64_t i = 0; i < entries_to_show; ++i) {
      AppendLine(kIndentPrefix, MetricPercent(category.entries[i]->metric), " ",
                 category.entries[i]->short_text);
    }
    const int64_t remaining_entries = category.entries.size() - entries_to_show;
    if (remaining_entries > 0) {
      AppendLine(kIndentPrefix, "... (", remaining_entries, " more ",
                 entry_name_, ")");
    }
  }
  const int64_t remaining_categories = categories.size() - categories_shown;
  if (remaining_categories > 0) {
    AppendTableRow(
        absl::StrCat("... (", remaining_categories, " more categories)"),
        expected_metric_sum_ - metric_sum, expected_metric_sum_);
  }
}

void MetricTableReport::AppendEntryTable() {
  AppendLine("********** ", entry_name_, " table for ", metric_name_,
             " **********");
  AppendLine();

  double metric_sum = UnaccountedMetric();
  int64_t entries_shown = 0;
  for (const auto& entry : entries_) {
    if (entries_shown >= max_entries_to_show_ ||
        metric_sum / expected_metric_sum_ > max_metric_proportion_to_show_) {
      break;
    }
    ++entries_shown;
    metric_sum += entry.metric;

    std::string text = entry.text;
    if (text.empty()) {
      text = "[no entry text]";
    }
    AppendTableRow(text, entry.metric, metric_sum);
  }
  const int64_t remaining_entries = entries_.size() - entries_shown;
  if (remaining_entries > 0) {
    AppendTableRow(
        absl::StrCat("... (", remaining_entries, " more ", entry_name_, ")"),
        expected_metric_sum_ - metric_sum, expected_metric_sum_);
  }
}

void MetricTableReport::AppendTableRow(const std::string& text,
                                       const double metric,
                                       const double running_metric_sum) {
  // This is the widest metric number possible, assuming non-negative metrics,
  // so align to that width.
  const int64_t max_metric_string_size =
      MetricString(expected_metric_sum_).size();
  std::string metric_string = MetricString(metric);

  // Don't try to make a gigantic string and crash if expected_metric_sum_ is
  // wrong somehow.
  int64_t padding_len = 1;
  const int64_t metric_string_size = metric_string.size();
  if (max_metric_string_size >= metric_string_size) {
    padding_len += max_metric_string_size - metric_string.size();
  }
  std::string padding(padding_len, ' ');
  AppendLine(padding, metric_string, " (", MetricPercent(metric), " Σ",
             MetricPercent(running_metric_sum), ")   ", text);
}

double MetricTableReport::UnaccountedMetric() {
  double metric_sum = 0.0;
  for (const auto& entry : entries_) {
    metric_sum += entry.metric;
  }
  return expected_metric_sum_ - metric_sum;
}

std::string MetricTableReport::MetricString(double metric) {
  // Round to integer and stringify.
  std::string s1 = absl::StrCat(std::llround(metric));

  // Code below commafies the string, e.g. "1234" becomes "1,234".
  absl::string_view sp1(s1);
  std::string output;
  // Copy leading non-digit characters unconditionally.
  // This picks up the leading sign.
  while (!sp1.empty() && !absl::ascii_isdigit(sp1[0])) {
    output.push_back(sp1[0]);
    sp1.remove_prefix(1);
  }
  // Copy rest of input characters.
  for (int64_t i = 0, end = sp1.size(); i < end; ++i) {
    if (i > 0 && (sp1.size() - i) % 3 == 0) {
      output.push_back(',');
    }
    output.push_back(sp1[i]);
  }
  return output;
}

std::string MetricTableReport::MetricPercent(double metric) {
  return absl::StrFormat("%5.2f%%", metric / expected_metric_sum_ * 100.0);
}

}  // namespace xla
