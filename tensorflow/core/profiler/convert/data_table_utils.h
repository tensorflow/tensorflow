
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
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "nlohmann/json_fwd.hpp"
#include "nlohmann/json.hpp"
namespace tensorflow {
namespace profiler {

static const char kBooleanTypeCode = 'B';
static const char kNumberTypeCode = 'N';
static const char kTextTypeCode = 'T';

class Value {
 public:
  explicit Value() : null_(false) {}

  // This type is neither copyable nor movable.
  Value(const Value&) = delete;
  Value& operator=(const Value&) = delete;
  virtual ~Value() = default;

  // Returns the type of this value
  virtual const char& GetType() const = 0;

  // Returns whether or not this cell's value is a logical null
  virtual bool IsNull() const { return null_; }

 protected:
  void set_null(bool null) { null_ = null; }

 private:
  bool null_;
};

class TextValue : public Value {
 public:
  TextValue() : value_("") { set_null(true); }  // A NULL TextValue.
  explicit TextValue(absl::string_view value) : value_(value) {}

  // This type is neither copyable nor movable.
  TextValue(const TextValue&) = delete;
  TextValue& operator=(const TextValue&) = delete;
  ~TextValue() override = default;

  // Returns the type of this value
  const char& GetType() const override { return kTextTypeCode; }

  // Returns the text value
  const std::string& GetValue() const {
    DCHECK(!IsNull()) << "This is a NULL value.";
    return value_;
  }

 private:
  std::string value_;
};

class NumberValue : public Value {
 public:
  NumberValue() : value_(0.0) { set_null(true); }  // A NULL NumberValue.
  explicit NumberValue(double value) : value_(value) {}

  // This type is neither copyable nor movable.
  NumberValue(const NumberValue&) = delete;
  NumberValue& operator=(const NumberValue&) = delete;
  ~NumberValue() override = default;

  // Returns the type of this value
  const char& GetType() const override { return kNumberTypeCode; }

  // Returns the number value
  double GetValue() const {
    DCHECK(!IsNull()) << "This is a NULL value.";
    return value_;
  }

 private:
  double value_;
};

class BooleanValue : public Value {
 public:
  BooleanValue() : value_(false) { set_null(true); }  // A NULL BooleanValue.
  explicit BooleanValue(bool value) : value_(value) {}

  // This type is neither copyable nor movable.
  BooleanValue(const BooleanValue&) = delete;
  BooleanValue& operator=(const BooleanValue&) = delete;
  ~BooleanValue() override = default;

  // Returns the type of this value
  const char& GetType() const override { return kBooleanTypeCode; }

  // Returns the boolean value
  bool GetValue() const {
    DCHECK(!IsNull()) << "This is a NULL value.";
    return value_;
  }

 private:
  bool value_;
};

// We now only support value type of TextValue, NumberValue and BooleanValue.
// We Don't deal with formatted values yet.
struct TableCell {
  // Constructors with one argument - the value.
  explicit TableCell(bool value) : value(new BooleanValue(value)) {}
  explicit TableCell(double value) : value(new NumberValue(value)) {}
  explicit TableCell(absl::string_view value) : value(new TextValue(value)) {}
  explicit TableCell(const char* value) : value(new TextValue(value)) {}

  // Constructors with two argument - value and custom properties.
  TableCell(Value* value,
            const absl::btree_map<std::string, std::string>& custom_properties)
      : value(value), custom_properties(custom_properties) {}
  TableCell(bool value,
            const absl::btree_map<std::string, std::string>& custom_properties)
      : value(new BooleanValue(value)), custom_properties(custom_properties) {}
  TableCell(double value,
            const absl::btree_map<std::string, std::string>& custom_properties)
      : value(new NumberValue(value)), custom_properties(custom_properties) {}
  TableCell(absl::string_view value,
            const absl::btree_map<std::string, std::string>& custom_properties)
      : value(new TextValue(value)), custom_properties(custom_properties) {}
  TableCell(const char* value,
            const absl::btree_map<std::string, std::string>& custom_properties)
      : value(new TextValue(value)), custom_properties(custom_properties) {}

  nlohmann::json GetCellValue() const { return GetCellValueImp(value.get()); }

  nlohmann::json GetCellValueImp(const Value* value) const {
    if (value == nullptr || value->IsNull()) {
      return "";
    }
    const char type_code = value->GetType();
    if (type_code == kTextTypeCode) {
      auto text_value = dynamic_cast<const TextValue*>(value);
      if (text_value != nullptr) {
        return text_value->GetValue();
      }
    }
    if (type_code == kNumberTypeCode) {
      auto number_value = dynamic_cast<const NumberValue*>(value);
      if (number_value != nullptr) {
        return number_value->GetValue();
      }
    }
    if (type_code == kBooleanTypeCode) {
      auto boolean_value = dynamic_cast<const BooleanValue*>(value);
      if (boolean_value != nullptr) {
        return boolean_value->GetValue();
      }
    }
    return "";
  }

  std::string GetCellValueStr() const {
    return GetCellValueStrImp(value.get());
  }

  std::string GetCellValueStrImp(const Value* value) const {
    if (value == nullptr || value->IsNull()) {
      return "";
    }
    const char type_code = value->GetType();
    if (type_code == kTextTypeCode) {
      auto text_value = dynamic_cast<const TextValue*>(value);
      if (text_value != nullptr) {
        return text_value->GetValue();
      }
    }
    if (type_code == kNumberTypeCode) {
      auto number_value = dynamic_cast<const NumberValue*>(value);
      if (number_value != nullptr) {
        return absl::StrCat(absl::LegacyPrecision(number_value->GetValue()));
      }
    }
    if (type_code == kBooleanTypeCode) {
      auto boolean_value = dynamic_cast<const BooleanValue*>(value);
      if (boolean_value != nullptr) {
        // Don't use StrCat here as it converts bools to "0" or "1".
        return boolean_value->GetValue() ? "TRUE" : "FALSE";
      }
    }
    return "";
  }
  std::unique_ptr<Value> value;
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
  TableRow& AddNumberCell(double value) {
    cells_.push_back(std::make_unique<TableCell>(value));
    return *this;
  }
  TableRow& AddTextCell(absl::string_view value) {
    cells_.push_back(std::make_unique<TableCell>(value));
    return *this;
  }
  TableRow& AddBooleanCell(bool value) {
    cells_.push_back(std::make_unique<TableCell>(value));
    return *this;
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
        cell_json["v"] = cell->GetCellValue();
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
