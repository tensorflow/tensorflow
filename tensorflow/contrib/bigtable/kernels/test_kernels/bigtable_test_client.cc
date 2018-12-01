/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "google/bigtable/v2/data.pb.h"
#include "google/protobuf/wrappers.pb.h"
#include "re2/re2.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/ptr_util.h"
// #include "util/task/codes.pb.h"

namespace tensorflow {
namespace {

void UpdateRow(const ::google::bigtable::v2::Mutation& mut,
               std::map<string, string>* row) {
  if (mut.has_set_cell()) {
    CHECK(mut.set_cell().timestamp_micros() >= -1)
        << "Timestamp_micros: " << mut.set_cell().timestamp_micros();
    auto col =
        strings::Printf("%s:%s", mut.set_cell().family_name().c_str(),
                        string(mut.set_cell().column_qualifier()).c_str());
    (*row)[col] = string(mut.set_cell().value());
  } else if (mut.has_delete_from_column()) {
    auto col = strings::Printf(
        "%s:%s", mut.delete_from_column().family_name().c_str(),
        string(mut.delete_from_column().column_qualifier()).c_str());
    row->erase(col);
  } else if (mut.has_delete_from_family()) {
    auto itr = row->lower_bound(mut.delete_from_family().family_name());
    auto prefix =
        strings::Printf("%s:", mut.delete_from_family().family_name().c_str());
    while (itr != row->end() && itr->first.substr(0, prefix.size()) == prefix) {
      row->erase(itr);
    }
  } else if (mut.has_delete_from_row()) {
    row->clear();
  } else {
    LOG(ERROR) << "Unknown mutation: " << mut.ShortDebugString();
  }
}

}  // namespace

class SampleRowKeysResponse : public grpc::ClientReaderInterface<
                                  google::bigtable::v2::SampleRowKeysResponse> {
 public:
  explicit SampleRowKeysResponse(BigtableTestClient* client)
      : client_(client) {}

  bool NextMessageSize(uint32_t* sz) override {
    mutex_lock l(mu_);
    mutex_lock l2(client_->mu_);
    if (num_messages_sent_ * 2 < client_->table_.rows.size()) {
      *sz = 10000;  // A sufficiently high enough value to not worry about.
      return true;
    }
    return false;
  }

  bool Read(google::bigtable::v2::SampleRowKeysResponse* resp) override {
    // Send every other key from the table.
    mutex_lock l(mu_);
    mutex_lock l2(client_->mu_);
    *resp = google::bigtable::v2::SampleRowKeysResponse();
    auto itr = client_->table_.rows.begin();
    for (uint64 i = 0; i < 2 * num_messages_sent_; ++i) {
      ++itr;
      if (itr == client_->table_.rows.end()) {
        return false;
      }
    }
    resp->set_row_key(itr->first);
    resp->set_offset_bytes(100 * num_messages_sent_);
    num_messages_sent_++;
    return true;
  }

  grpc::Status Finish() override { return grpc::Status::OK; }

  void WaitForInitialMetadata() override {}  // Do nothing.

 private:
  mutex mu_;
  int64 num_messages_sent_ GUARDED_BY(mu_) = 0;
  BigtableTestClient* client_;  // Not owned.
};

class ReadRowsResponse : public grpc::ClientReaderInterface<
                             google::bigtable::v2::ReadRowsResponse> {
 public:
  ReadRowsResponse(BigtableTestClient* client,
                   google::bigtable::v2::ReadRowsRequest const& request)
      : client_(client), request_(request) {}

  bool NextMessageSize(uint32_t* sz) override {
    mutex_lock l(mu_);
    if (sent_first_message_) {
      return false;
    }
    *sz = 10000000;  // A sufficiently high enough value to not worry about.
    return true;
  }

  bool Read(google::bigtable::v2::ReadRowsResponse* resp) override {
    mutex_lock l(mu_);
    if (sent_first_message_) {
      return false;
    }
    sent_first_message_ = true;
    RowFilter filter = MakeRowFilter();

    mutex_lock l2(client_->mu_);
    *resp = google::bigtable::v2::ReadRowsResponse();
    // Send all contents in first response.
    for (auto itr = client_->table_.rows.begin();
         itr != client_->table_.rows.end(); ++itr) {
      if (filter.AllowRow(itr->first)) {
        ::google::bigtable::v2::ReadRowsResponse_CellChunk* chunk = nullptr;
        bool sent_first = false;
        for (auto col_itr = itr->second.columns.begin();
             col_itr != itr->second.columns.end(); ++col_itr) {
          if (filter.AllowColumn(col_itr->first)) {
            chunk = resp->add_chunks();
            if (!sent_first) {
              sent_first = true;
              chunk->set_row_key(itr->first);
            }
            auto colon_idx = col_itr->first.find(":");
            CHECK(colon_idx != string::npos)
                << "No ':' found in: " << col_itr->first;
            chunk->mutable_family_name()->set_value(
                string(col_itr->first, 0, colon_idx));
            chunk->mutable_qualifier()->set_value(
                string(col_itr->first, ++colon_idx));
            if (!filter.strip_values) {
              chunk->set_value(col_itr->second);
            }
            if (filter.only_one_column) {
              break;
            }
          }
        }
        if (sent_first) {
          // We are sending this row, so set the commit flag on the last chunk.
          chunk->set_commit_row(true);
        }
      }
    }
    return true;
  }

  grpc::Status Finish() override { return grpc::Status::OK; }

  void WaitForInitialMetadata() override {}  // Do nothing.

 private:
  struct RowFilter {
    std::set<string> row_set;
    std::vector<std::pair<string, string>> row_ranges;
    double row_sample = 0.0;  // Note: currently ignored.
    std::unique_ptr<RE2> col_filter;
    bool strip_values = false;
    bool only_one_column = false;

    bool AllowRow(const string& row) {
      if (row_set.find(row) != row_set.end()) {
        return true;
      }
      for (const auto& range : row_ranges) {
        if (range.first <= row && range.second > row) {
          return true;
        }
      }
      return false;
    }

    bool AllowColumn(const string& col) {
      if (col_filter) {
        return RE2::FullMatch(col, *col_filter);
      } else {
        return true;
      }
    }
  };

  RowFilter MakeRowFilter() {
    RowFilter filter;
    for (auto i = request_.rows().row_keys().begin();
         i != request_.rows().row_keys().end(); ++i) {
      filter.row_set.insert(string(*i));
    }
    for (auto i = request_.rows().row_ranges().begin();
         i != request_.rows().row_ranges().end(); ++i) {
      if (i->start_key_case() !=
              google::bigtable::v2::RowRange::kStartKeyClosed ||
          i->end_key_case() != google::bigtable::v2::RowRange::kEndKeyOpen) {
        LOG(WARNING) << "Skipping row range that cannot be processed: "
                     << i->ShortDebugString();
        continue;
      }
      filter.row_ranges.emplace_back(std::make_pair(
          string(i->start_key_closed()), string(i->end_key_open())));
    }
    if (request_.filter().has_chain()) {
      string family_filter;
      string qualifier_filter;
      for (auto i = request_.filter().chain().filters().begin();
           i != request_.filter().chain().filters().end(); ++i) {
        switch (i->filter_case()) {
          case google::bigtable::v2::RowFilter::kFamilyNameRegexFilter:
            family_filter = i->family_name_regex_filter();
            break;
          case google::bigtable::v2::RowFilter::kColumnQualifierRegexFilter:
            qualifier_filter = i->column_qualifier_regex_filter();
            break;
          case google::bigtable::v2::RowFilter::kCellsPerColumnLimitFilter:
            if (i->cells_per_column_limit_filter() != 1) {
              LOG(ERROR) << "Unexpected cells_per_column_limit_filter: "
                         << i->cells_per_column_limit_filter();
            }
            break;
          case google::bigtable::v2::RowFilter::kStripValueTransformer:
            filter.strip_values = i->strip_value_transformer();
            break;
          case google::bigtable::v2::RowFilter::kRowSampleFilter:
            LOG(INFO) << "Ignoring row sample directive.";
            break;
          case google::bigtable::v2::RowFilter::kPassAllFilter:
            break;
          case google::bigtable::v2::RowFilter::kCellsPerRowLimitFilter:
            filter.only_one_column = true;
            break;
          default:
            LOG(WARNING) << "Ignoring unknown filter type: "
                         << i->ShortDebugString();
        }
      }
      if (family_filter.empty() || qualifier_filter.empty()) {
        LOG(WARNING) << "Missing regex!";
      } else {
        string regex = strings::Printf("%s:%s", family_filter.c_str(),
                                       qualifier_filter.c_str());
        filter.col_filter.reset(new RE2(regex));
      }
    } else {
      LOG(WARNING) << "Read request did not have a filter chain specified: "
                   << request_.filter().DebugString();
    }
    return filter;
  }

  mutex mu_;
  bool sent_first_message_ GUARDED_BY(mu_) = false;
  BigtableTestClient* client_;  // Not owned.
  const google::bigtable::v2::ReadRowsRequest request_;
};

class MutateRowsResponse : public grpc::ClientReaderInterface<
                               google::bigtable::v2::MutateRowsResponse> {
 public:
  explicit MutateRowsResponse(size_t num_successes)
      : num_successes_(num_successes) {}

  bool NextMessageSize(uint32_t* sz) override {
    mutex_lock l(mu_);
    if (sent_first_message_) {
      return false;
    }
    *sz = 10000000;  // A sufficiently high enough value to not worry about.
    return true;
  }

  bool Read(google::bigtable::v2::MutateRowsResponse* resp) override {
    mutex_lock l(mu_);
    if (sent_first_message_) {
      return false;
    }
    sent_first_message_ = true;
    *resp = google::bigtable::v2::MutateRowsResponse();
    for (size_t i = 0; i < num_successes_; ++i) {
      auto entry = resp->add_entries();
      entry->set_index(i);
    }
    return true;
  }

  grpc::Status Finish() override { return grpc::Status::OK; }

  void WaitForInitialMetadata() override {}  // Do nothing.

 private:
  const size_t num_successes_;

  mutex mu_;
  bool sent_first_message_ = false;
};

grpc::Status BigtableTestClient::MutateRow(
    grpc::ClientContext* context,
    google::bigtable::v2::MutateRowRequest const& request,
    google::bigtable::v2::MutateRowResponse* response) {
  mutex_lock l(mu_);
  auto* row = &table_.rows[string(request.row_key())];
  for (int i = 0; i < request.mutations_size(); ++i) {
    UpdateRow(request.mutations(i), &row->columns);
  }
  *response = google::bigtable::v2::MutateRowResponse();
  return grpc::Status::OK;
}
grpc::Status BigtableTestClient::CheckAndMutateRow(
    grpc::ClientContext* context,
    google::bigtable::v2::CheckAndMutateRowRequest const& request,
    google::bigtable::v2::CheckAndMutateRowResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                      "CheckAndMutateRow not implemented.");
}
grpc::Status BigtableTestClient::ReadModifyWriteRow(
    grpc::ClientContext* context,
    google::bigtable::v2::ReadModifyWriteRowRequest const& request,
    google::bigtable::v2::ReadModifyWriteRowResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                      "ReadModifyWriteRow not implemented.");
}
std::unique_ptr<
    grpc::ClientReaderInterface<google::bigtable::v2::ReadRowsResponse>>
BigtableTestClient::ReadRows(
    grpc::ClientContext* context,
    google::bigtable::v2::ReadRowsRequest const& request) {
  return MakeUnique<ReadRowsResponse>(this, request);
}

std::unique_ptr<
    grpc::ClientReaderInterface<google::bigtable::v2::SampleRowKeysResponse>>
BigtableTestClient::SampleRowKeys(
    grpc::ClientContext* context,
    google::bigtable::v2::SampleRowKeysRequest const& request) {
  return MakeUnique<SampleRowKeysResponse>(this);
}
std::unique_ptr<
    grpc::ClientReaderInterface<google::bigtable::v2::MutateRowsResponse>>
BigtableTestClient::MutateRows(
    grpc::ClientContext* context,
    google::bigtable::v2::MutateRowsRequest const& request) {
  mutex_lock l(mu_);
  for (auto i = request.entries().begin(); i != request.entries().end(); ++i) {
    auto* row = &table_.rows[string(i->row_key())];
    for (auto mut = i->mutations().begin(); mut != i->mutations().end();
         ++mut) {
      UpdateRow(*mut, &row->columns);
    }
  }
  return MakeUnique<MutateRowsResponse>(request.entries_size());
}

std::unique_ptr<grpc::ClientAsyncResponseReaderInterface<
    google::bigtable::v2::MutateRowResponse>>
BigtableTestClient::AsyncMutateRow(
    grpc::ClientContext* context,
    google::bigtable::v2::MutateRowRequest const& request,
    grpc::CompletionQueue* cq) {
  LOG(WARNING) << "Call to InMemoryDataClient::" << __func__
               << "(); this will likely cause a crash!";
  return nullptr;
}

std::unique_ptr<::grpc::ClientAsyncReaderInterface<
    ::google::bigtable::v2::SampleRowKeysResponse>>
BigtableTestClient::AsyncSampleRowKeys(
    ::grpc::ClientContext* context,
    const ::google::bigtable::v2::SampleRowKeysRequest& request,
    ::grpc::CompletionQueue* cq, void* tag) {
  LOG(WARNING) << "Call to InMemoryDataClient::" << __func__
               << "(); this will likely cause a crash!";
  return nullptr;
}

std::unique_ptr<::grpc::ClientAsyncReaderInterface<
    ::google::bigtable::v2::MutateRowsResponse>>
BigtableTestClient::AsyncMutateRows(
    ::grpc::ClientContext* context,
    const ::google::bigtable::v2::MutateRowsRequest& request,
    ::grpc::CompletionQueue* cq, void* tag) {
  LOG(WARNING) << "Call to InMemoryDataClient::" << __func__
               << "(); this will likely cause a crash!";
  return nullptr;
}

std::shared_ptr<grpc::Channel> BigtableTestClient::Channel() {
  LOG(WARNING) << "Call to InMemoryDataClient::Channel(); this will likely "
                  "cause a crash!";
  return nullptr;
}
}  // namespace tensorflow
