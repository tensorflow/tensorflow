/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/bigtable/kernels/test_kernels/bigtable_test_client.h"
#include "google/cloud/bigtable/internal/table.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

void WriteCell(const string& row, const string& family, const string& column,
               const string& value,
               ::google::cloud::bigtable::noex::Table* table) {
  ::google::cloud::bigtable::SingleRowMutation mut(row);
  mut.emplace_back(::google::cloud::bigtable::SetCell(family, column, value));
  table->Apply(std::move(mut));
}

TEST(BigtableTestClientTest, EmptyRowRead) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  ::google::cloud::bigtable::RowSet rowset;
  rowset.Append("r1");
  auto filter = ::google::cloud::bigtable::Filter::Chain(
      ::google::cloud::bigtable::Filter::Latest(1));
  auto rows = table.ReadRows(std::move(rowset), filter);
  EXPECT_EQ(rows.begin(), rows.end()) << "Some rows were returned in response!";
}

TEST(BigtableTestClientTest, SingleRowWriteAndRead) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);

  ::google::cloud::bigtable::RowSet rowset("r1");
  auto filter = ::google::cloud::bigtable::Filter::Chain(
      ::google::cloud::bigtable::Filter::Latest(1));
  auto rows = table.ReadRows(std::move(rowset), filter);
  auto itr = rows.begin();
  EXPECT_NE(itr, rows.end()) << "No rows were returned in response!";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r1");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v1");

  ++itr;
  EXPECT_EQ(itr, rows.end());
}

TEST(BigtableTestClientTest, MultiRowWriteAndSingleRowRead) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);
  WriteCell("r2", "f1", "c1", "v2", &table);
  WriteCell("r3", "f1", "c1", "v3", &table);

  ::google::cloud::bigtable::RowSet rowset("r1");
  auto filter = ::google::cloud::bigtable::Filter::Chain(
      ::google::cloud::bigtable::Filter::Latest(1));
  auto rows = table.ReadRows(std::move(rowset), filter);
  auto itr = rows.begin();

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r1");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v1");

  ++itr;
  EXPECT_EQ(itr, rows.end()) << "Extra rows in the response.";
}

TEST(BigtableTestClientTest, MultiRowWriteAndRead) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);
  WriteCell("r2", "f1", "c1", "v2", &table);
  WriteCell("r3", "f1", "c1", "v3", &table);

  ::google::cloud::bigtable::RowSet rowset("r1", "r2", "r3");
  auto filter = ::google::cloud::bigtable::Filter::Chain(
      ::google::cloud::bigtable::Filter::Latest(1));
  auto rows = table.ReadRows(std::move(rowset), filter);
  auto itr = rows.begin();

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r1");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v1");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r2");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v2");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r3");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v3");

  ++itr;
  EXPECT_EQ(itr, rows.end()) << "Extra rows in the response.";
}

TEST(BigtableTestClientTest, MultiRowWriteAndPrefixRead) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);
  WriteCell("r2", "f1", "c1", "v2", &table);
  WriteCell("r3", "f1", "c1", "v3", &table);

  auto filter = ::google::cloud::bigtable::Filter::Chain(
      ::google::cloud::bigtable::Filter::Latest(1));
  auto rows =
      table.ReadRows(::google::cloud::bigtable::RowRange::Prefix("r"), filter);
  auto itr = rows.begin();

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r1");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v1");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r2");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v2");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r3");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v3");

  ++itr;
  EXPECT_EQ(itr, rows.end()) << "Extra rows in the response.";
}

TEST(BigtableTestClientTest, ColumnFiltering) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);
  WriteCell("r2", "f1", "c1", "v2", &table);
  WriteCell("r3", "f1", "c1", "v3", &table);

  // Extra cells
  WriteCell("r1", "f2", "c1", "v1", &table);
  WriteCell("r2", "f2", "c1", "v2", &table);
  WriteCell("r3", "f1", "c2", "v3", &table);

  auto filter = ::google::cloud::bigtable::Filter::Chain(
      ::google::cloud::bigtable::Filter::Latest(1),
      ::google::cloud::bigtable::Filter::FamilyRegex("f1"),
      ::google::cloud::bigtable::Filter::ColumnRegex("c1"));
  auto rows =
      table.ReadRows(::google::cloud::bigtable::RowRange::Prefix("r"), filter);
  auto itr = rows.begin();

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r1");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v1");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r2");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v2");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r3");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "v3");

  ++itr;
  EXPECT_EQ(itr, rows.end()) << "Extra rows in the response.";
}

TEST(BigtableTestClientTest, RowKeys) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);
  WriteCell("r2", "f1", "c1", "v2", &table);
  WriteCell("r3", "f1", "c1", "v3", &table);

  // Extra cells
  WriteCell("r1", "f2", "c1", "v1", &table);
  WriteCell("r2", "f2", "c1", "v2", &table);
  WriteCell("r3", "f1", "c2", "v3", &table);

  auto filter = ::google::cloud::bigtable::Filter::Chain(
      ::google::cloud::bigtable::Filter::Latest(1),
      ::google::cloud::bigtable::Filter::CellsRowLimit(1),
      ::google::cloud::bigtable::Filter::StripValueTransformer());
  auto rows =
      table.ReadRows(::google::cloud::bigtable::RowRange::Prefix("r"), filter);
  auto itr = rows.begin();
  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r1");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r2");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "");

  ++itr;

  EXPECT_NE(itr, rows.end()) << "Missing rows";
  EXPECT_TRUE(*itr) << "Error reading row: " << itr->status().message();
  EXPECT_EQ((*itr)->row_key(), "r3");
  EXPECT_EQ((*itr)->cells().size(), 1);
  EXPECT_EQ((*itr)->cells()[0].family_name(), "f1");
  EXPECT_EQ((*itr)->cells()[0].column_qualifier(), "c1");
  EXPECT_EQ((*itr)->cells()[0].value(), "");

  ++itr;
  EXPECT_EQ(itr, rows.end()) << "Extra rows in the response.";
}

TEST(BigtableTestClientTest, SampleKeys) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);
  WriteCell("r2", "f1", "c1", "v2", &table);
  WriteCell("r3", "f1", "c1", "v3", &table);
  WriteCell("r4", "f1", "c1", "v4", &table);
  WriteCell("r5", "f1", "c1", "v5", &table);

  grpc::Status status;
  auto resp = table.SampleRows(status);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(3, resp.size());
  EXPECT_EQ("r1", string(resp[0].row_key));
  EXPECT_EQ(0, resp[0].offset_bytes);
  EXPECT_EQ("r3", string(resp[1].row_key));
  EXPECT_EQ(100, resp[1].offset_bytes);
  EXPECT_EQ("r5", string(resp[2].row_key));
  EXPECT_EQ(200, resp[2].offset_bytes);
}

TEST(BigtableTestClientTest, SampleKeysShort) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);

  grpc::Status status;
  auto resp = table.SampleRows(status);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(1, resp.size());
  EXPECT_EQ("r1", string(resp[0].row_key));
}

TEST(BigtableTestClientTest, SampleKeysEvenNumber) {
  std::shared_ptr<::google::cloud::bigtable::DataClient> client_ptr =
      std::make_shared<BigtableTestClient>();
  ::google::cloud::bigtable::noex::Table table(client_ptr, "test_table");

  WriteCell("r1", "f1", "c1", "v1", &table);
  WriteCell("r2", "f1", "c1", "v2", &table);
  WriteCell("r3", "f1", "c1", "v3", &table);
  WriteCell("r4", "f1", "c1", "v4", &table);

  grpc::Status status;
  auto resp = table.SampleRows(status);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(2, resp.size());
  EXPECT_EQ("r1", string(resp[0].row_key));
  EXPECT_EQ("r3", string(resp[1].row_key));
}

}  // namespace
}  // namespace tensorflow
