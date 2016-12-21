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

#include "tensorflow/core/kernels/cloud/bigquery_table_accessor.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/kernels/cloud/bigquery_table_accessor_test_data.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/cloud/http_request_fake.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

constexpr char kTestProject[] = "test-project";
constexpr char kTestDataset[] = "test-dataset";
constexpr char kTestTable[] = "test-table";

static bool HasSubstr(const string& base, const string& substr) {
  bool ok = StringPiece(base).contains(substr);
  EXPECT_TRUE(ok) << base << ", expected substring " << substr;
  return ok;
}

class FakeAuthProvider : public AuthProvider {
 public:
  Status GetToken(string* token) override {
    *token = "fake_token";
    return Status::OK();
  }
};

static string DeterministicSerialization(const tensorflow::Example& example) {
  const int size = example.ByteSize();
  string result(size, '\0');
  ::tensorflow::protobuf::io::ArrayOutputStream array_stream(
      gtl::string_as_array(&result), size);
  ::tensorflow::protobuf::io::CodedOutputStream output_stream(&array_stream);

  output_stream.SetSerializationDeterministic(true);
  example.SerializeWithCachedSizes(&output_stream);
  EXPECT_FALSE(output_stream.HadError());
  EXPECT_EQ(size, output_stream.ByteCount());
  return result;
}

}  // namespace

class BigQueryTableAccessorTest : public ::testing::Test {
 protected:
  BigQueryTableAccessor::SchemaNode GetSchema() {
    return accessor_->schema_root_;
  }

  Status CreateTableAccessor(const string& project_id, const string& dataset_id,
                             const string& table_id, int64 timestamp_millis,
                             int64 row_buffer_size,
                             const std::set<string>& columns,
                             const BigQueryTablePartition& partition) {
    return BigQueryTableAccessor::New(
        project_id, dataset_id, table_id, timestamp_millis, row_buffer_size,
        columns, partition, std::unique_ptr<AuthProvider>(new FakeAuthProvider),
        std::unique_ptr<HttpRequest::Factory>(
            new FakeHttpRequestFactory(&requests_)),
        &accessor_);
  }

  std::vector<HttpRequest*> requests_;
  std::unique_ptr<BigQueryTableAccessor> accessor_;
};

TEST_F(BigQueryTableAccessorTest, NegativeTimestamp) {
  const auto status =
      CreateTableAccessor(kTestProject, kTestDataset, kTestTable, -1, 3, {},
                          BigQueryTablePartition());
  EXPECT_TRUE(errors::IsInvalidArgument(status));
}

TEST_F(BigQueryTableAccessorTest, ZeroTimestamp) {
  const auto status =
      CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 0, 3, {},
                          BigQueryTablePartition());
  EXPECT_TRUE(errors::IsInvalidArgument(status));
}

TEST_F(BigQueryTableAccessorTest, RepeatedFieldNoAllowedTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      R"({
        "kind": "bigquery#table",
        "etag": "\"4zcX32ezvFoFzxHoG04qJqKZk6c/MTQ1Nzk3NTgwNzE4Mw\"",
        "id": "test-project:test-dataset.test-table",
        "schema": {
          "fields": [
          {
            "name": "int_field",
            "type": "INTEGER",
            "mode": "REPEATED"
          }]
        },
        "numRows": "10"
      })"));
  const auto status =
      CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 3, {},
                          BigQueryTablePartition());
  EXPECT_TRUE(errors::IsUnimplemented(status));
  EXPECT_TRUE(HasSubstr(status.error_message(),
                        "Tables with repeated columns are not supported"));
}

TEST_F(BigQueryTableAccessorTest, ValidSchemaTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      kSampleSchema));
  TF_EXPECT_OK(CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 3,
                                   {}, BigQueryTablePartition()));
  // Validate total number of rows.
  EXPECT_EQ(4, accessor_->total_num_rows());

  // Validate the schema.
  const auto schema_root = GetSchema();
  EXPECT_EQ(schema_root.name, "");
  EXPECT_EQ(schema_root.type, BigQueryTableAccessor::ColumnType::kNone);
  EXPECT_EQ(9, schema_root.schema_nodes.size());

  EXPECT_EQ(schema_root.schema_nodes[0].name, "int_field");
  EXPECT_EQ(schema_root.schema_nodes[0].type,
            BigQueryTableAccessor::ColumnType::kInteger);

  EXPECT_EQ(schema_root.schema_nodes[1].name, "str_field");
  EXPECT_EQ(schema_root.schema_nodes[1].type,
            BigQueryTableAccessor::ColumnType::kString);

  EXPECT_EQ(1, schema_root.schema_nodes[2].schema_nodes.size());
  EXPECT_EQ(schema_root.schema_nodes[2].name, "rec_field");
  EXPECT_EQ(schema_root.schema_nodes[2].type,
            BigQueryTableAccessor::ColumnType::kRecord);

  EXPECT_EQ(schema_root.schema_nodes[2].schema_nodes[0].name,
            "rec_field.float_field");
  EXPECT_EQ(schema_root.schema_nodes[2].schema_nodes[0].type,
            BigQueryTableAccessor::ColumnType::kFloat);

  EXPECT_EQ(schema_root.schema_nodes[3].name, "bool_field");
  EXPECT_EQ(schema_root.schema_nodes[3].type,
            BigQueryTableAccessor::ColumnType::kBoolean);

  EXPECT_EQ(schema_root.schema_nodes[4].name, "bytes_field");
  EXPECT_EQ(schema_root.schema_nodes[4].type,
            BigQueryTableAccessor::ColumnType::kBytes);

  EXPECT_EQ(schema_root.schema_nodes[5].name, "timestamp_field");
  EXPECT_EQ(schema_root.schema_nodes[5].type,
            BigQueryTableAccessor::ColumnType::kTimestamp);

  EXPECT_EQ(schema_root.schema_nodes[6].name, "date_field");
  EXPECT_EQ(schema_root.schema_nodes[6].type,
            BigQueryTableAccessor::ColumnType::kDate);

  EXPECT_EQ(schema_root.schema_nodes[7].name, "time_field");
  EXPECT_EQ(schema_root.schema_nodes[7].type,
            BigQueryTableAccessor::ColumnType::kTime);

  EXPECT_EQ(schema_root.schema_nodes[8].name, "datetime_field");
  EXPECT_EQ(schema_root.schema_nodes[8].type,
            BigQueryTableAccessor::ColumnType::kDatetime);
}

TEST_F(BigQueryTableAccessorTest, ReadOneRowTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      kSampleSchema));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/data?maxResults=1&startIndex=2\n"
      "Auth Token: fake_token\n",
      kTestRow));
  BigQueryTablePartition partition;
  partition.set_start_index(2);
  partition.set_end_index(3);
  TF_EXPECT_OK(CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 1,
                                   {}, partition));
  int64 row_id;
  Example example;
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));

  // Validate returned result.
  Example expected_example;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(kTestExampleProto,
                                                    &expected_example));
  EXPECT_EQ(DeterministicSerialization(expected_example),
            DeterministicSerialization(example));
  EXPECT_EQ(row_id, 2);
  EXPECT_TRUE(accessor_->Done());
}

TEST_F(BigQueryTableAccessorTest, ReadOneRowPartialTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      kSampleSchema));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/data?maxResults=1&startIndex=2\n"
      "Auth Token: fake_token\n",
      kTestRow));
  BigQueryTablePartition partition;
  partition.set_start_index(2);
  partition.set_end_index(3);
  TF_EXPECT_OK(CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 1,
                                   {"bool_field", "rec_field.float_field"},
                                   partition));
  int64 row_id;
  Example example;
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));

  // Validate returned result.
  EXPECT_EQ(row_id, 2);
  EXPECT_TRUE(accessor_->Done());
  Example expected_example;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(kTestPartialExampleProto,
                                                    &expected_example));
  EXPECT_EQ(DeterministicSerialization(expected_example),
            DeterministicSerialization(example));
}

TEST_F(BigQueryTableAccessorTest, ReadOneRowWithNullsTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      kSampleSchema));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/data?maxResults=1&startIndex=2\n"
      "Auth Token: fake_token\n",
      kTestRowWithNulls));
  BigQueryTablePartition partition;
  partition.set_start_index(2);
  partition.set_end_index(3);
  TF_EXPECT_OK(CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 1,
                                   {}, partition));
  int64 row_id;
  Example example;
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));

  // Validate returned result.
  Example expected_example;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(kTestExampleProtoWithNulls,
                                                    &expected_example));
  EXPECT_EQ(DeterministicSerialization(expected_example),
            DeterministicSerialization(example));
  EXPECT_EQ(row_id, 2);
  EXPECT_TRUE(accessor_->Done());
}

TEST_F(BigQueryTableAccessorTest, BrokenRowTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      kSampleSchema));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/data?maxResults=1&startIndex=2\n"
      "Auth Token: fake_token\n",
      kBrokenTestRow));
  BigQueryTablePartition partition;
  partition.set_start_index(2);
  partition.set_end_index(3);
  TF_EXPECT_OK(CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 1,
                                   {}, partition));
  int64 row_id;
  Example example;
  const auto status = accessor_->ReadRow(&row_id, &example);
  EXPECT_TRUE(errors::IsInternal(status));
  EXPECT_TRUE(
      HasSubstr(status.error_message(), "Cannot convert value to integer"));
}

TEST_F(BigQueryTableAccessorTest, MultiplePagesTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      kSampleSchema));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/data?maxResults=2&startIndex=1\n"
      "Auth Token: fake_token\n",
      kTestTwoRows));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/"
      "data?maxResults=2&pageToken=next_page\n"
      "Auth Token: fake_token\n",
      kTestRowWithNulls));

  BigQueryTablePartition partition;
  partition.set_start_index(1);
  partition.set_end_index(-1);
  TF_EXPECT_OK(CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 2,
                                   {}, partition));

  int64 row_id;
  Example example;
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));
  EXPECT_EQ(1, row_id);
  EXPECT_FALSE(accessor_->Done());
  EXPECT_EQ(
      (example.features().feature()).at("int_field").int64_list().value(0),
      1111);

  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));
  EXPECT_EQ(2, row_id);
  EXPECT_FALSE(accessor_->Done());
  EXPECT_EQ(example.features().feature().at("int_field").int64_list().value(0),
            2222);

  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));
  EXPECT_EQ(3, row_id);
  EXPECT_TRUE(accessor_->Done());
  Example expected_example;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(kTestExampleProtoWithNulls,
                                                    &expected_example));
  EXPECT_EQ(DeterministicSerialization(expected_example),
            DeterministicSerialization(example));
  EXPECT_TRUE(errors::IsOutOfRange(accessor_->ReadRow(&row_id, &example)));
}

TEST_F(BigQueryTableAccessorTest, SwitchingPartitionsTest) {
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/\n"
      "Auth Token: fake_token\n",
      kSampleSchema));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/data?maxResults=2&startIndex=0\n"
      "Auth Token: fake_token\n",
      kTestTwoRows));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/"
      "data?maxResults=2&startIndex=3\n"
      "Auth Token: fake_token\n",
      kTestRowWithNulls));
  requests_.emplace_back(new FakeHttpRequest(
      "Uri: https://www.googleapis.com/bigquery/v2/projects/test-project/"
      "datasets/test-dataset/tables/test-table/data?maxResults=2&startIndex=0\n"
      "Auth Token: fake_token\n",
      kTestTwoRows));

  BigQueryTablePartition partition;
  partition.set_start_index(0);
  partition.set_end_index(1);
  TF_EXPECT_OK(CreateTableAccessor(kTestProject, kTestDataset, kTestTable, 1, 2,
                                   {}, partition));

  int64 row_id;
  Example example;
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));
  EXPECT_EQ(0, row_id);
  EXPECT_TRUE(accessor_->Done());
  EXPECT_EQ(example.features().feature().at("int_field").int64_list().value(0),
            1111);

  partition.set_start_index(3);
  partition.set_end_index(-1);
  accessor_->SetPartition(partition);
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));
  EXPECT_EQ(3, row_id);
  EXPECT_TRUE(accessor_->Done());
  EXPECT_EQ(example.features().feature().at("int_field").int64_list().value(0),
            1234);

  partition.set_start_index(0);
  partition.set_end_index(2);
  accessor_->SetPartition(partition);
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));
  EXPECT_EQ(0, row_id);
  EXPECT_FALSE(accessor_->Done());
  EXPECT_EQ(example.features().feature().at("int_field").int64_list().value(0),
            1111);
  TF_EXPECT_OK(accessor_->ReadRow(&row_id, &example));
  EXPECT_EQ(1, row_id);
  EXPECT_TRUE(accessor_->Done());
  EXPECT_EQ(example.features().feature().at("int_field").int64_list().value(0),
            2222);
}

}  // namespace tensorflow
