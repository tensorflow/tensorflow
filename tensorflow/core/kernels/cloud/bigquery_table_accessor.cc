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
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

namespace {

constexpr size_t kBufferSize = 1024 * 1024;  // In bytes.

Status ParseJson(StringPiece json, Json::Value* result) {
  Json::Reader reader;
  if (!reader.parse(json.ToString(), *result)) {
    return errors::Internal("Couldn't parse JSON response from BigQuery.");
  }
  return Status::OK();
}

string ColumnTypeToString(BigQueryTableAccessor::ColumnType enum_type) {
  switch (enum_type) {
    case BigQueryTableAccessor::ColumnType::kRecord:
      return "RECORD";
    case BigQueryTableAccessor::ColumnType::kString:
      return "STRING";
    case BigQueryTableAccessor::ColumnType::kBytes:
      return "BYTES";
    case BigQueryTableAccessor::ColumnType::kInteger:
      return "INTEGER";
    case BigQueryTableAccessor::ColumnType::kFloat:
      return "FLOAT";
    case BigQueryTableAccessor::ColumnType::kBoolean:
      return "BOOLEAN";
    case BigQueryTableAccessor::ColumnType::kTimestamp:
      return "TIMESTAMP";
    case BigQueryTableAccessor::ColumnType::kDate:
      return "DATE";
    case BigQueryTableAccessor::ColumnType::kTime:
      return "TIME";
    case BigQueryTableAccessor::ColumnType::kDatetime:
      return "DATETIME";
    case BigQueryTableAccessor::ColumnType::kNone:
      return "NONE";
  }
}

Status ParseColumnType(const string& type,
                       BigQueryTableAccessor::ColumnType* enum_type) {
  if (type == "RECORD") {
    *enum_type = BigQueryTableAccessor::ColumnType::kRecord;
  } else if (type == "STRING") {
    *enum_type = BigQueryTableAccessor::ColumnType::kString;
  } else if (type == "BYTES") {
    *enum_type = BigQueryTableAccessor::ColumnType::kBytes;
  } else if (type == "INTEGER") {
    *enum_type = BigQueryTableAccessor::ColumnType::kInteger;
  } else if (type == "FLOAT") {
    *enum_type = BigQueryTableAccessor::ColumnType::kFloat;
  } else if (type == "BOOLEAN") {
    *enum_type = BigQueryTableAccessor::ColumnType::kBoolean;
  } else if (type == "TIMESTAMP") {
    *enum_type = BigQueryTableAccessor::ColumnType::kTimestamp;
  } else if (type == "DATE") {
    *enum_type = BigQueryTableAccessor::ColumnType::kDate;
  } else if (type == "TIME") {
    *enum_type = BigQueryTableAccessor::ColumnType::kTime;
  } else if (type == "DATETIME") {
    *enum_type = BigQueryTableAccessor::ColumnType::kDatetime;
  } else {
    return errors::Internal(
        strings::StrCat("Could not parse column type ", type));
  }
  return Status::OK();
}

}  // namespace

Status BigQueryTableAccessor::New(
    const string& project_id, const string& dataset_id, const string& table_id,
    int64 timestamp_millis, int64 row_buffer_size,
    const std::set<string>& columns, const BigQueryTablePartition& partition,
    std::unique_ptr<BigQueryTableAccessor>* accessor) {
  return New(project_id, dataset_id, table_id, timestamp_millis,
             row_buffer_size, columns, partition, nullptr, nullptr, accessor);
}

Status BigQueryTableAccessor::New(
    const string& project_id, const string& dataset_id, const string& table_id,
    int64 timestamp_millis, int64 row_buffer_size,
    const std::set<string>& columns, const BigQueryTablePartition& partition,
    std::unique_ptr<AuthProvider> auth_provider,
    std::unique_ptr<HttpRequest::Factory> http_request_factory,
    std::unique_ptr<BigQueryTableAccessor>* accessor) {
  if (timestamp_millis <= 0) {
    return errors::InvalidArgument(
        "Cannot use zero or negative timestamp to query a table.");
  }
  if (auth_provider == nullptr && http_request_factory == nullptr) {
    accessor->reset(new BigQueryTableAccessor(project_id, dataset_id, table_id,
                                              timestamp_millis, row_buffer_size,
                                              columns, partition));
  } else {
    accessor->reset(new BigQueryTableAccessor(
        project_id, dataset_id, table_id, timestamp_millis, row_buffer_size,
        columns, partition, std::move(auth_provider),
        std::move(http_request_factory)));
  }
  return (*accessor)->ReadSchema();
}

BigQueryTableAccessor::BigQueryTableAccessor(
    const string& project_id, const string& dataset_id, const string& table_id,
    int64 timestamp_millis, int64 row_buffer_size,
    const std::set<string>& columns, const BigQueryTablePartition& partition)
    : BigQueryTableAccessor(
          project_id, dataset_id, table_id, timestamp_millis, row_buffer_size,
          columns, partition,
          std::unique_ptr<AuthProvider>(new GoogleAuthProvider()),
          std::unique_ptr<HttpRequest::Factory>(new HttpRequest::Factory())) {
  row_buffer_.resize(row_buffer_size);
}

BigQueryTableAccessor::BigQueryTableAccessor(
    const string& project_id, const string& dataset_id, const string& table_id,
    int64 timestamp_millis, int64 row_buffer_size,
    const std::set<string>& columns, const BigQueryTablePartition& partition,
    std::unique_ptr<AuthProvider> auth_provider,
    std::unique_ptr<HttpRequest::Factory> http_request_factory)
    : project_id_(project_id),
      dataset_id_(dataset_id),
      table_id_(table_id),
      timestamp_millis_(timestamp_millis),
      columns_(columns),
      partition_(partition),
      auth_provider_(std::move(auth_provider)),
      http_request_factory_(std::move(http_request_factory)) {
  row_buffer_.resize(row_buffer_size);
  Reset();
}

void BigQueryTableAccessor::SetPartition(
    const BigQueryTablePartition& partition) {
  partition_ = partition;
  Reset();
}

void BigQueryTableAccessor::Reset() {
  first_buffered_row_index_ = partition_.start_index();
  next_row_in_buffer_ = -1;
  next_page_token_ = "";
}

Status BigQueryTableAccessor::ReadRow(int64* row_id, Example* example) {
  if (Done()) {
    return errors::OutOfRange("Reached end of table ", FullTableName());
  }

  // If the next row is already fetched and cached, return the row from the
  // buffer. Otherwise, fill up the row buffer from BigQuery and return a row.
  if (next_row_in_buffer_ != -1 && next_row_in_buffer_ < row_buffer_.size()) {
    *row_id = first_buffered_row_index_ + next_row_in_buffer_;
    *example = row_buffer_[next_row_in_buffer_];
    next_row_in_buffer_++;
  } else {
    string auth_token;
    TF_RETURN_IF_ERROR(
        AuthProvider::GetToken(auth_provider_.get(), &auth_token));

    std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
    std::vector<char> output_buffer;
    output_buffer.reserve(kBufferSize);
    TF_RETURN_IF_ERROR(request->Init());

    // The first time that we access BigQuery there is no page token. After that
    // we use the page token (which returns rows faster).
    if (!next_page_token_.empty()) {
      TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
          BigQueryUriPrefix(), "data?maxResults=", row_buffer_.size(),
          "&pageToken=", request->EscapeString(next_page_token_))));
      first_buffered_row_index_ += row_buffer_.size();
    } else {
      TF_RETURN_IF_ERROR(request->SetUri(strings::StrCat(
          BigQueryUriPrefix(), "data?maxResults=", row_buffer_.size(),
          "&startIndex=", first_buffered_row_index_)));
    }
    TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
    TF_RETURN_IF_ERROR(request->SetResultBuffer(&output_buffer));
    TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading rows from ",
                                    FullTableName());

    // Parse the returned row.
    StringPiece response_piece =
        StringPiece(&output_buffer[0], output_buffer.size());
    Json::Value root;
    TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));
    for (unsigned int i = 0; i < root["rows"].size(); ++i) {
      row_buffer_[i].Clear();
      TF_RETURN_IF_ERROR(
          ParseColumnValues(root["rows"][i], schema_root_, &row_buffer_[i]));
    }

    next_page_token_ = root["pageToken"].asString();
    *row_id = first_buffered_row_index_;
    *example = row_buffer_[0];
    next_row_in_buffer_ = 1;
  }
  return Status::OK();
}

Status BigQueryTableAccessor::ParseColumnValues(
    const Json::Value& value, const SchemaNode& root_schema_node,
    Example* example) {
  if (value.empty()) {
    return Status::OK();
  }
  if (value["f"].isNull()) {
    return Status::OK();
  }
  int value_index = 0;
  for (const auto& schema_node : root_schema_node.schema_nodes) {
    if (value["f"][value_index].isNull()) {
      value_index++;
      continue;
    }

    if (schema_node.type == ColumnType::kRecord) {
      TF_RETURN_IF_ERROR(ParseColumnValues(value["f"][value_index]["v"],
                                           schema_node, example));
    } else {
      // Append the column value only if user has requested the column.
      if (columns_.empty() ||
          columns_.find(schema_node.name) != columns_.end()) {
        TF_RETURN_IF_ERROR(AppendValueToExample(schema_node.name,
                                                value["f"][value_index]["v"],
                                                schema_node.type, example));
      }
    }
    value_index++;
  }
  return Status::OK();
}

Status BigQueryTableAccessor::ReadSchema() {
  string auth_token;
  TF_RETURN_IF_ERROR(AuthProvider::GetToken(auth_provider_.get(), &auth_token));

  // Send a request to read the schema.
  std::unique_ptr<HttpRequest> request(http_request_factory_->Create());
  std::vector<char> output_buffer;
  output_buffer.reserve(kBufferSize);
  TF_RETURN_IF_ERROR(request->Init());
  TF_RETURN_IF_ERROR(request->SetUri(BigQueryUriPrefix()));
  TF_RETURN_IF_ERROR(request->AddAuthBearerHeader(auth_token));
  TF_RETURN_IF_ERROR(request->SetResultBuffer(&output_buffer));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(request->Send(), " when reading schema for ",
                                  FullTableName());

  // Parse the schema.
  StringPiece response_piece =
      StringPiece(&output_buffer[0], output_buffer.size());

  Json::Value root;
  TF_RETURN_IF_ERROR(ParseJson(response_piece, &root));
  const auto& columns = root["schema"]["fields"];
  string column_name_prefix = "";
  schema_root_ = {"", ColumnType::kNone};
  TF_RETURN_IF_ERROR(
      ExtractColumnType(columns, column_name_prefix, &schema_root_));
  if (root["numRows"].isNull()) {
    return errors::Internal("Number of rows cannot be extracted for table ",
                            FullTableName());
  }
  strings::safe_strto64(root["numRows"].asString().c_str(), &total_num_rows_);
  return Status::OK();
}

Status BigQueryTableAccessor::ExtractColumnType(
    const Json::Value& columns, const string& column_name_prefix,
    SchemaNode* root) {
  for (auto columns_it = columns.begin(); columns_it != columns.end();
       ++columns_it) {
    if ((*columns_it)["mode"].asString() == "REPEATED") {
      return errors::Unimplemented(strings::StrCat(
          "Tables with repeated columns are not supported: ", FullTableName()));
    }
    ColumnType type;
    const string current_column_name = strings::StrCat(
        column_name_prefix, (*columns_it)["name"].asString().c_str());
    TF_RETURN_IF_ERROR(
        ParseColumnType((*columns_it)["type"].asString().c_str(), &type));
    root->schema_nodes.emplace_back(current_column_name, type);
    if (type == ColumnType::kRecord) {
      const auto new_prefix = strings::StrCat(current_column_name, ".");
      TF_RETURN_IF_ERROR(ExtractColumnType((*columns_it)["fields"], new_prefix,
                                           &root->schema_nodes.back()));
    }
  }
  return Status::OK();
}

Status BigQueryTableAccessor::AppendValueToExample(
    const string& column_name, const Json::Value& column_value,
    const BigQueryTableAccessor::ColumnType type, Example* example) {
  if (column_value.isNull()) {
    return Status::OK();
  }
  auto& feature =
      (*example->mutable_features()->mutable_feature())[column_name];

  switch (type) {
    case BigQueryTableAccessor::ColumnType::kNone:
    case BigQueryTableAccessor::ColumnType::kRecord:
      return errors::Unimplemented("Cannot append type to an example.");
    case BigQueryTableAccessor::ColumnType::kTimestamp:
    case BigQueryTableAccessor::ColumnType::kDate:
    case BigQueryTableAccessor::ColumnType::kTime:
    case BigQueryTableAccessor::ColumnType::kDatetime:
    case BigQueryTableAccessor::ColumnType::kString:
    case BigQueryTableAccessor::ColumnType::kBytes:
      feature.mutable_bytes_list()->add_value(column_value.asString());
      break;
    case BigQueryTableAccessor::ColumnType::kBoolean:
      feature.mutable_int64_list()->add_value(
          column_value.asString() == "false" ? 0 : 1);
      break;
    case BigQueryTableAccessor::ColumnType::kInteger:
      int64 column_value_int64;
      if (!strings::safe_strto64(column_value.asString().c_str(),
                                 &column_value_int64)) {
        return errors::Internal("Cannot convert value to integer ",
                                column_value.asString().c_str());
      }
      feature.mutable_int64_list()->add_value(column_value_int64);
      break;
    case BigQueryTableAccessor::ColumnType::kFloat:
      // BigQuery float is actually a double.
      double column_value_double;
      if (!strings::safe_strtod(column_value.asString().c_str(),
                                &column_value_double)) {
        return errors::Internal("Cannot convert value to double: ",
                                column_value.asString().c_str());
      }
      feature.mutable_float_list()->add_value(
          static_cast<float>(column_value_double));
      break;
  }
  return Status::OK();
}

string BigQueryTableAccessor::BigQueryTableAccessor::BigQueryUriPrefix() {
  HttpRequest request;
  return strings::StrCat("https://www.googleapis.com/bigquery/v2/projects/",
                         request.EscapeString(project_id_), "/datasets/",
                         request.EscapeString(dataset_id_), "/tables/",
                         request.EscapeString(table_id_), "/");
}

string BigQueryTableAccessor::FullTableName() {
  return strings::StrCat(project_id_, ":", dataset_id_, ".", table_id_, "@",
                         timestamp_millis_);
}

bool BigQueryTableAccessor::Done() {
  return (total_num_rows_ <= first_buffered_row_index_ + next_row_in_buffer_) ||
         (partition_.end_index() != -1 &&
          partition_.end_index() <=
              first_buffered_row_index_ + next_row_in_buffer_);
}

}  // namespace tensorflow
