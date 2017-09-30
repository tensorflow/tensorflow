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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CLOUD_BIGQUERY_PARTITION_ACCESSOR_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CLOUD_BIGQUERY_PARTITION_ACCESSOR_H_

#include <map>
#include <memory>
#include <vector>

#include "tensorflow/contrib/cloud/kernels/bigquery_table_partition.pb.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/cloud/curl_http_request.h"
#include "tensorflow/core/platform/cloud/google_auth_provider.h"

namespace tensorflow {

/// This class facilitates accessing BigQuery tables.
///
/// Notes:
///  - Nested fields are not supported.
///  - BigQuery 'Record's are automatically flattened,
///  - BigQuery float type is a double but is converted to a C++ float in this
///    class.
///  - It is possible for a table snapshot to go out-of-scope in the BigQuery
///    service while accessing the table if a very old timestamp is used. For
///    exact details, see 'Table Decorators' in BigQuery docs.
class BigQueryTableAccessor {
 public:
  // Column types supported by BigQuery.
  enum class ColumnType {
    kString = 0,
    kBytes,
    kInteger,
    kFloat,
    kBoolean,
    kTimestamp,
    kDate,
    kTime,
    kDatetime,
    kRecord,
    kNone
  };

  /// \brief Creates a new BigQueryTableAccessor object.
  //
  // We do not allow relative (negative or zero) snapshot times here since we
  // want to have a consistent snapshot of the table for the lifetime of this
  // object.
  // Use end_point if you want to connect to a different end point than the
  // official BigQuery end point. Otherwise send an empty string.
  static Status New(const string& project_id, const string& dataset_id,
                    const string& table_id, int64 timestamp_millis,
                    int64 row_buffer_size, const string& end_point,
                    const std::vector<string>& columns,
                    const BigQueryTablePartition& partition,
                    std::unique_ptr<BigQueryTableAccessor>* accessor);

  /// \brief Starts reading a new partition.
  Status SetPartition(const BigQueryTablePartition& partition);

  /// \brief Returns true if there are more rows available in the current
  /// partition.
  bool Done();

  /// \brief Returns a single row as example proto.
  ///
  /// This function will return an error if the table snapshot goes out of scope
  /// in the BigQuery service.
  Status ReadRow(int64* row_id, Example* example);

  /// \brief Returns total number of rows in the table.
  int64 total_num_rows() { return total_num_rows_; }

  virtual ~BigQueryTableAccessor() {}

 private:
  friend class BigQueryTableAccessorTest;

  // This struct encapsulates schema nodes for a BigQuery table.
  struct SchemaNode {
    SchemaNode() {}
    SchemaNode(const string& name, ColumnType type) : name(name), type(type) {}

    string name;
    ColumnType type;
    std::vector<SchemaNode> schema_nodes;
  };

  /// If nullptr is passed for http_request_factory and auth_provider the
  /// default production ones are used. This can be used by tests to override
  /// these two variables.
  static Status New(const string& project_id, const string& dataset_id,
                    const string& table_id, int64 timestamp_millis,
                    int64 row_buffer_size, const string& end_point,
                    const std::vector<string>& columns,
                    const BigQueryTablePartition& partition,
                    std::unique_ptr<AuthProvider> auth_provider,
                    std::unique_ptr<HttpRequest::Factory> http_request_factory,
                    std::unique_ptr<BigQueryTableAccessor>* accessor);

  /// \brief Constructs an object for a given table and partition.
  BigQueryTableAccessor(const string& project_id, const string& dataset_id,
                        const string& table_id, int64 timestamp_millis,
                        int64 row_buffer_size, const string& end_point,
                        const std::vector<string>& columns,
                        const BigQueryTablePartition& partition);

  /// Used for unit testing.
  BigQueryTableAccessor(
      const string& project_id, const string& dataset_id,
      const string& table_id, int64 timestamp_millis, int64 row_buffer_size,
      const string& end_point, const std::vector<string>& columns,
      const BigQueryTablePartition& partition,
      std::unique_ptr<AuthProvider> auth_provider,
      std::unique_ptr<HttpRequest::Factory> http_request_factory);

  /// \brief Parses column values for a given row.
  Status ParseColumnValues(const Json::Value& value,
                           const SchemaNode& root_schema_node,
                           Example* example);

  /// \brief Reads the table schema and stores it.
  Status ReadSchema();

  /// \brief Extracts column type from a column in schema.
  Status ExtractColumnType(const Json::Value& columns,
                           const string& column_name_prefix, SchemaNode* root);

  /// \brief Appends a single BigQuery column Value to 'example' for a given
  /// column.
  Status AppendValueToExample(const string& column_name,
                              const Json::Value& column_value,
                              const BigQueryTableAccessor::ColumnType type,
                              Example* example);

  /// \brief Resets internal counters for reading a partition.
  void Reset();

  /// \brief Helper function that returns BigQuery http endpoint prefix.
  string BigQueryUriPrefix();

  /// \brief Computes the maxResults arg to send to BigQuery.
  int64 ComputeMaxResultsArg();

  /// \brief Returns full name of the underlying table name.
  string FullTableName() {
    return strings::StrCat(project_id_, ":", dataset_id_, ".", table_id_, "@",
                           timestamp_millis_);
  }

  const string project_id_;
  const string dataset_id_;
  const string table_id_;

  // Snapshot timestamp.
  const int64 timestamp_millis_;

  // Columns that should be read. Empty means all columns.
  const std::set<string> columns_;

  // HTTP address of BigQuery end point to use.
  const string bigquery_end_point_;

  // Describes the portion of the table that we are currently accessing.
  BigQueryTablePartition partition_;

  // Total number of rows in the underlying table.
  int64 total_num_rows_ = 0;

  // Offset of the first row in the underlying row_buffer_.
  int64 first_buffered_row_index_ = 0;

  // Offset of the next row in the row_buffer_. -1 indicates that this index
  // is invalid.
  int next_row_in_buffer_ = -1;

  // This buffer holds next rows to improve performance. Its size will be
  // based on how much buffering was requested.
  std::vector<Example> row_buffer_;

  // If next_page is set, it will used to read next batch of data.
  string next_page_token_;

  // A tree representing the schema for the underlying table.
  SchemaNode schema_root_;

  std::unique_ptr<AuthProvider> auth_provider_;
  std::unique_ptr<HttpRequest::Factory> http_request_factory_;

  TF_DISALLOW_COPY_AND_ASSIGN(BigQueryTableAccessor);
};

}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_CLOUD_BIGQUERY_PARTITION_ACCESSOR_H_
