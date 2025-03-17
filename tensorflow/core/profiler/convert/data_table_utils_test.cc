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

#include "tensorflow/core/profiler/convert/data_table_utils.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "nlohmann/json_fwd.hpp"
#include "nlohmann/json.hpp"

namespace tensorflow::profiler {
namespace {

std::vector<std::vector<std::string>> GetTestColumns() {
  return {{"rank", "number", "Rank"},
          {"program_id", "string", "Program Id"},
          {"op_category", "string", "Op Category"},
          {"op_name", "string", "Op Name"},
          {"bytes_accessed", "number", "Bytes Accessed"},
          {"model_flops", "number", "Model Flops"},
          {"occurrences", "number", "#Occurrences"}};
}

std::vector<nlohmann::json> GetTestRows() {
  return {{1, "11111", "category1", "op1", 200000000, 123123123, 10},
          {2, "22222", "category2", "op2", 1000000, 0, 20},
          {3, "33333", "category3", "op3", 3000000, 565656, 30}};
}

std::unique_ptr<tensorflow::profiler::DataTable> CreateTestDataTable() {
  auto data_table = std::make_unique<DataTable>();
  for (const std::vector<std::string>& col : GetTestColumns()) {
    data_table->AddColumn(TableColumn(col[0], col[1], col[2]));
  }
  for (const nlohmann::json& row_json : GetTestRows()) {
    TableRow* row = data_table->AddRow();
    for (int i = 0; i < row_json.size(); ++i) {
      row->AddCell(row_json[i]);
    }
  }
  return data_table;
}

std::unique_ptr<tensorflow::profiler::DataTable>
CreateTestDataTableWithCustomProperties() {
  auto data_table = std::make_unique<DataTable>();
  data_table->AddCustomProperty("key1", "value1");
  data_table->AddCustomProperty("key2", "value2");
  return data_table;
}

TEST(DataTableUtilsTest, ToJson) {
  std::unique_ptr<tensorflow::profiler::DataTable> data_table =
      CreateTestDataTable();
  std::string json_string = data_table->ToJson();
  const nlohmann::basic_json<> parsed_json = nlohmann::json::parse(json_string);
  auto test_columns = GetTestColumns();
  auto test_rows = GetTestRows();
  EXPECT_EQ(parsed_json["cols"].size(), test_columns.size());
  EXPECT_EQ(parsed_json["rows"].size(), test_rows.size());
  for (int i = 0; i < test_columns.size(); ++i) {
    EXPECT_EQ(parsed_json["cols"][i]["id"], test_columns[i][0]);
    EXPECT_EQ(parsed_json["cols"][i]["label"], test_columns[i][2]);
    EXPECT_EQ(parsed_json["cols"][i]["type"], test_columns[i][1]);
  }
  for (int i = 0; i < test_rows.size(); ++i) {
    for (int j = 0; j < test_columns.size(); ++j) {
      EXPECT_EQ(parsed_json["rows"][i]["c"][j]["v"], GetTestRows()[i][j]);
    }
  }
}

TEST(DataTableUtilsTest, ToJsonWithCustomProperties) {
  std::unique_ptr<tensorflow::profiler::DataTable> data_table =
      CreateTestDataTableWithCustomProperties();
  std::string table_json_string = data_table->ToJson();
  const nlohmann::basic_json<> parsed_json =
      nlohmann::json::parse(table_json_string);
  EXPECT_EQ(parsed_json.find("p")->size(), 2);
  EXPECT_EQ(parsed_json.find("p")->at("key1"), "value1");
  EXPECT_EQ(parsed_json.find("p")->at("key2"), "value2");
}

}  // namespace
}  // namespace tensorflow::profiler
