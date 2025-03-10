
/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_DATA_TABLE_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_DATA_TABLE_UTILS_H_
#include <memory>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/strings/str_replace.h"
#include "nlohmann/json_fwd.hpp"
#include "nlohmann/json.hpp"
namespace tensorflow {
namespace profiler {
// We Don't deal with formatted values on backend now.
struct TableCell {
  TableCell() = default;
  explicit TableCell(nlohmann::json value) : value(value) {};
  explicit TableCell(
      nlohmann::json value,
      absl::btree_map<std::string, std::string> custom_properties)
      : value(value), custom_properties(custom_properties) {};
  std::string value_str() const {
    return absl::StrReplaceAll(value.dump(), {{"\"", ""}});
  }
  nlohmann::json value;
  absl::btree_map<std::string, std::string> custom_properties;
};
struct TableColumn {
  TableColumn() = default;
  explicit TableColumn(std::string id, std::string type, std::string label)
      : id(id), type(type), label(label) {};
  explicit TableColumn(
      std::string id, std::string type, std::string label,
      absl::btree_map<std::string, std::string> custom_properties)
      : id(id), type(type), label(label), custom_properties(custom_properties) {
        };
  std::string id;
  std::string type;
  std::string label;
  absl::btree_map<std::string, std::string> custom_properties;
};
class TableRow {
 public:
  TableRow() = default;
  virtual ~TableRow() = default;
  // Adds a value of a single cell to the end of the row.
  // Memory will be freed by the TableRow.
  TableCell* AddCell(nlohmann::json value) {
    cells_.push_back(std::make_unique<TableCell>(value));
    return cells_.back().get();
  }
  std::vector<const TableCell*> GetCells() const {
    std::vector<const TableCell*> cells;
    cells.reserve(cells_.size());
    for (const std::unique_ptr<TableCell>& cell : cells_) {
      cells.push_back(cell.get());
    }
    return cells;
  }
  void SetCustomProperties(
      const absl::btree_map<std::string, std::string>& custom_properties) {
    custom_properties_ = custom_properties;
  }
  void AddCustomProperty(std::string name, std::string value) {
    custom_properties_[name] = value;
  }
  const absl::btree_map<std::string, std::string>& GetCustomProperties() const {
    return custom_properties_;
  }
  int RowSize() const { return cells_.size(); }

 private:
  std::vector<std::unique_ptr<TableCell>> cells_;
  absl::btree_map<std::string, std::string> custom_properties_;
};
// A DataTable class that can be used to create a DataTable JSON/CSV
// serialization. We need this class instead raw JSON manipulation because we
// need to support custom properties.
class DataTable {
 public:
  DataTable() = default;
  void AddColumn(TableColumn column) { table_descriptions_.push_back(column); }
  const std::vector<TableColumn>& GetColumns() { return table_descriptions_; }
  // Create an empty row and return a pointer to it.
  // DataTable takes the ownership of the returned TableRow.
  TableRow* AddRow() {
    table_rows_.push_back(std::make_unique<TableRow>());
    return table_rows_.back().get();
  }
  std::vector<const TableRow*> GetRows() {
    std::vector<const TableRow*> rows;
    rows.reserve(table_rows_.size());
    for (const std::unique_ptr<TableRow>& row : table_rows_) {
      rows.push_back(row.get());
    }
    return rows;
  }
  void AddCustomProperty(std::string name, std::string value) {
    custom_properties_[name] = value;
  }
  std::string ToJson() {
    nlohmann::json table;
    table["cols"] = nlohmann::json::array();
    table["rows"] = nlohmann::json::array();
    if (!custom_properties_.empty()) {
      table["p"] = custom_properties_;
    }
    for (const TableColumn& col : table_descriptions_) {
      nlohmann::json column_json;
      column_json["id"] = col.id;
      column_json["type"] = col.type;
      column_json["label"] = col.label;
      if (!col.custom_properties.empty()) {
        column_json["p"] = col.custom_properties;
      }
      table["cols"].push_back(column_json);
    }
    for (const std::unique_ptr<TableRow>& row : table_rows_) {
      nlohmann::json row_json;
      row_json["c"] = nlohmann::json::array();
      for (const TableCell* cell : row->GetCells()) {
        nlohmann::json cell_json;
        cell_json["v"] = cell->value;
        if (!cell->custom_properties.empty()) {
          cell_json["p"] = cell->custom_properties;
        }
        row_json["c"].push_back(cell_json);
      }
      if (!row->GetCustomProperties().empty()) {
        row_json["p"] = row->GetCustomProperties();
      }
      table["rows"].push_back(row_json);
    }
    return table.dump();
  }

 private:
  std::vector<TableColumn> table_descriptions_;
  std::vector<std::unique_ptr<TableRow>> table_rows_;
  absl::btree_map<std::string, std::string> custom_properties_;
};
}  // namespace profiler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_DATA_TABLE_UTILS_H_
