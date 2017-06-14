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

#include <cctype>
#include <unordered_map>

#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

void MetricTableReport::AddEntry(Entry entry) {
  entries_.push_back(std::move(entry));
}

void MetricTableReport::SetMetricName(string metric_name) {
  metric_name_ = std::move(metric_name);
}

void MetricTableReport::SetEntryName(string entry_name) {
  entry_name_ = std::move(entry_name);
}

void MetricTableReport::SetShowAllEntries() {
  max_entries_to_show_ = std::numeric_limits<int64>::max();
  max_entries_per_category_to_show_ = std::numeric_limits<int64>::max();
  max_metric_proportion_to_show_ = 1.1;  // more than 100%
}

void MetricTableReport::SetShowCategoryTable() { show_category_table_ = true; }

void MetricTableReport::SetShowEntryTable() { show_entry_table_ = true; }

string MetricTableReport::MakeReport(double expected_metric_sum) {
  expected_metric_sum_ = expected_metric_sum;
  report_.clear();

  // Sort the entries.
  const auto metric_greater = [](const Entry& a, const Entry& b) {
    return a.metric > b.metric;
  };
  std::sort(entries_.begin(), entries_.end(), metric_greater);

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

  int64 pos = 0;
  const string report = MakeReport(expected_metric_sum);
  while (pos < report.size()) {
    int64 end_of_line = report.find('\n', pos);
    if (end_of_line == string::npos) {
      end_of_line = report.size();
    }
    tensorflow::StringPiece line(report.data() + pos, end_of_line - pos);

    // TODO(b/34779244): Figure out how to do this without the verbose log-line
    // prefix. The usual way didn't compile on open source.
    LOG(INFO) << line;

    pos = end_of_line + 1;
  }
}

std::vector<MetricTableReport::Category> MetricTableReport::MakeCategories(
    const std::vector<Entry>* entries) {
  // Create the categories using a category_text -> category map.
  std::unordered_map<string, Category> category_map;
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
  std::sort(categories.begin(), categories.end(), metric_sum_greater);

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

  AppendLine("********** categories table **********");
  AppendLine("The left hand side numbers are ", metric_name_, ".");
  AppendLine();

  double metric_sum = UnaccountedMetric();
  int64 categories_shown = 0;
  for (const auto& category : categories) {
    if (categories_shown >= max_entries_to_show_ ||
        metric_sum / expected_metric_sum_ > max_metric_proportion_to_show_) {
      break;
    }
    ++categories_shown;
    metric_sum += category.metric_sum;

    // Show the category.
    string text = category.category_text;
    if (text.empty()) {
      text = "[no category]";
    }
    tensorflow::strings::StrAppend(&text, " (", category.entries.size(), " ",
                                   entry_name_, ")");
    AppendTableRow(text, category.metric_sum, metric_sum);

    // Show the top entries in the category.
    const char* const kIndentPrefix = "                              * ";
    int64 entries_to_show = std::min<int64>(max_entries_per_category_to_show_,
                                            category.entries.size());
    if (category.entries.size() == entries_to_show + 1) {
      // May as well show the last entry on the line that would otherwise say
      // that there is a single entry not shown.
      ++entries_to_show;
    }
    for (int64 i = 0; i < entries_to_show; ++i) {
      AppendLine(kIndentPrefix, MetricPercent(category.entries[i]->metric), " ",
                 category.entries[i]->short_text);
    }
    const int64 remaining_entries = category.entries.size() - entries_to_show;
    if (remaining_entries > 0) {
      AppendLine(kIndentPrefix, "... (", remaining_entries, " more ",
                 entry_name_, ")");
    }
  }
  const int64 remaining_categories = categories.size() - categories_shown;
  if (remaining_categories > 0) {
    AppendTableRow(tensorflow::strings::StrCat("... (", remaining_categories,
                                               " more categories)"),
                   expected_metric_sum_ - metric_sum, expected_metric_sum_);
  }
}

void MetricTableReport::AppendEntryTable() {
  AppendLine("********** ", entry_name_, " table **********");
  AppendLine("The left hand side numbers are ", metric_name_, ".");
  AppendLine();

  double metric_sum = UnaccountedMetric();
  int64 entries_shown = 0;
  for (const auto& entry : entries_) {
    if (entries_shown >= max_entries_to_show_ ||
        metric_sum / expected_metric_sum_ > max_metric_proportion_to_show_) {
      break;
    }
    ++entries_shown;
    metric_sum += entry.metric;

    string text = entry.text;
    if (text.empty()) {
      text = "[no entry text]";
    }
    AppendTableRow(text, entry.metric, metric_sum);
  }
  const int64 remaining_entries = entries_.size() - entries_shown;
  if (remaining_entries > 0) {
    AppendTableRow(tensorflow::strings::StrCat("... (", remaining_entries,
                                               " more ", entry_name_, ")"),
                   expected_metric_sum_ - metric_sum, expected_metric_sum_);
  }
}

void MetricTableReport::AppendTableRow(const string& text, const double metric,
                                       const double running_metric_sum) {
  // This is the widest metric number possible, assuming non-negative metrics,
  // so align to that width.
  const int64 max_metric_string_size =
      MetricString(expected_metric_sum_).size();
  string metric_string = MetricString(metric);

  // Don't try to make a gigantic string and crash if expected_metric_sum_ is
  // wrong somehow.
  int64 padding_len = 1;
  if (max_metric_string_size >= metric_string.size()) {
    padding_len += max_metric_string_size - metric_string.size();
  }
  string padding(padding_len, ' ');
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

string MetricTableReport::MetricString(double metric) {
  // Round to integer and stringify.
  string s1 = tensorflow::strings::StrCat(std::llround(metric));

  // Code below commafies the string, e.g. "1234" becomes "1,234".
  tensorflow::StringPiece sp1(s1);
  string output;
  // Copy leading non-digit characters unconditionally.
  // This picks up the leading sign.
  while (!sp1.empty() && !isdigit(sp1[0])) {
    output.push_back(sp1[0]);
    sp1.remove_prefix(1);
  }
  // Copy rest of input characters.
  for (int64 i = 0; i < sp1.size(); ++i) {
    if (i > 0 && (sp1.size() - i) % 3 == 0) {
      output.push_back(',');
    }
    output.push_back(sp1[i]);
  }
  return output;
}

string MetricTableReport::MetricPercent(double metric) {
  return tensorflow::strings::Printf("%5.2f%%",
                                     metric / expected_metric_sum_ * 100.0);
}

}  // namespace xla
