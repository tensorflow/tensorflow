/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_METRIC_TABLE_REPORT_H_
#define XLA_METRIC_TABLE_REPORT_H_

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"

namespace xla {

// Class for creating a text format table showing entries with a metric
// (e.g. cycles) and a text (e.g. name of function taking that many
// cycles). Entries are grouped by a category and sorted in decreasing order of
// the metric.
//
// Example of a categories table generated using this class:
//
// ********** microseconds report **********
// There are 3,912,517 microseconds in total.
// There are 123 microseconds ( 0.00%) not accounted for by the data.
// There are 3002 ops.
//
// ********** categories table **********
// The left hand side numbers are microseconds.
// 1,749,414 (44.71% Σ44.72%)   convolution (206 ops)
//                             * 10.51% %convolution.202
//                             * 10.51% %convolution.204
//                             * 10.51% %convolution.203
//                             * ... (203 more ops)
//   884,939 (22.62% Σ67.33%)   convolution window-dilated (7 ops)
//                             *  7.50% %convolution-window-dilated.7
// [...]
//
// The entry table is similar, it just has the entries directly as the entries
// instead of grouping by categories first.
class MetricTableReport {
 public:
  // Represents an entry in the table.
  struct Entry {
    // Text to show in the entry table for this entry.
    std::string text;

    // Text to show in the category table for this entry.
    std::string short_text;

    // Text that represents the category of this entry - entries with the same
    // category are grouped together in the category table.
    std::string category_text;

    // The value of the metric for this entry.
    double metric = 0.0;
  };

  void AddEntry(Entry entry);

  // The default name for the metric is "units", this function allows setting a
  // more meaningful name.
  void SetMetricName(std::string metric_name);

  // The default name for referring to entries is "entries", this functions
  // allows setting a more meaningful name.
  void SetEntryName(std::string entry_name);

  // By default the size of the table is limited. Calling this function forces
  // all entries to be shown no matter how many there are.
  void SetShowAllEntries();

  // Set option to show a table with data on the categories of entries.
  void SetShowCategoryTable();

  // Set option to show a table with data on the entries.
  void SetShowEntryTable();

  // Returns the report as a string. expected_metric_sum is the expected sum of
  // the metric across the entries. It is not an error for the actual sum to be
  // different from the expectation - the report will include the
  // discrepancy. All metric percentages are for the ratio with respect to the
  // expected sum, not the actual sum.
  std::string MakeReport(double expected_metric_sum);

  // As MakeReport(), but writes the report to the INFO log in a way that avoids
  // cutting the report off if it is longer than the maximum line length for a
  // logged line. Individual lines in the report may still be cut off, but they
  // would have to be very long for that to happen.
  void WriteReportToInfoLog(double expected_metric_sum);

 private:
  static constexpr double kDefaultMaxMetricProportionToShow = 0.99;
  static constexpr int64_t kDefaultMaxEntriesToShow = 100;
  static constexpr int64_t kDefaultMaxEntriesPerCategoryToShow = 5;

  // Append all parameters to the report.
  template <typename... Args>
  void AppendLine(Args... args) {
    absl::StrAppend(&report_, std::forward<Args>(args)..., "\n");
  }

  // Represents a set of entries with the same category_text.
  struct Category {
    std::string category_text;
    double metric_sum = 0.0;  // Sum of metric across entries.
    std::vector<const Entry*> entries;
  };

  // Returns a vector of categories of entries with the same category_text. The
  // vector is sorted in order of decreasing metric sum.
  //
  // The returned categories contain pointers into the entries parameter. The
  // style guide requires parameters to which references/pointers are retained
  // to be taken by pointer, even for const parameters, so that is why entries
  // is taken by pointer.
  static std::vector<Category> MakeCategories(
      const std::vector<Entry>* entries);

  // Append a header to the report.
  void AppendHeader();

  // Append a table of categories to the report.
  void AppendCategoryTable();

  // Append a table of entries to the report.
  void AppendEntryTable();

  // Appends a row of a table to the report.
  void AppendTableRow(const std::string& text, const double metric,
                      const double running_metric_sum);

  // Returns the discrepancy between the expected sum of the metric of the
  // entries and the actual sum.
  double UnaccountedMetric();

  // Formats the metric value as a string.
  std::string MetricString(double metric);

  // Returns a string representing the metric value as a proportion of the
  // expected metric sum.
  std::string MetricPercent(double metric);

  // The entries to make a report about.
  std::vector<Entry> entries_;

  double expected_metric_sum_ = 0.0;
  std::string metric_name_ = "units";
  std::string entry_name_ = "entries";
  bool show_category_table_ = false;
  bool show_entry_table_ = false;

  // These members control how many categories and entries to show in tables.
  int64_t max_entries_to_show_ = kDefaultMaxEntriesToShow;
  int64_t max_entries_per_category_to_show_ =
      kDefaultMaxEntriesPerCategoryToShow;
  double max_metric_proportion_to_show_ = kDefaultMaxMetricProportionToShow;

  // The report that is being created.
  std::string report_;
};

}  // namespace xla

#endif  // XLA_METRIC_TABLE_REPORT_H_
