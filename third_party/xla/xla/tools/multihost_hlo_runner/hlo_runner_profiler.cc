/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tools/multihost_hlo_runner/hlo_runner_profiler.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/lib/profiler_passes.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

constexpr absl::string_view kAllReduceErrorToken = "ERROR";

// Multi-pass profiler implementation for single-host HLO runner execution.
class MultiPassHloProfiler : public HLORunnerProfiler {
 public:
  MultiPassHloProfiler(absl::string_view dump_path, bool keep_xspace)
      : HLORunnerProfiler(dump_path, keep_xspace) {}

  void CreateSession() override {
    multi_passes_ =
        tsl::ProfilerPasses::Create(tsl::ProfilerPasses::DefaultOptions());
  }

  bool IsMultipassEnabled() const override { return true; }

  absl::StatusOr<bool> NeedMorePasses() override {
    if (multi_passes_ == nullptr) {
      return false;
    }
    return multi_passes_->NeedMorePasses();
  }

  void StartPass() override {
    if (multi_passes_ != nullptr) {
      absl::Status status = multi_passes_->StartPass();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to start pass: " << status;
      }
    }
  }

  void PushRange(absl::string_view name) override {
    if (multi_passes_ != nullptr) {
      absl::Status status = multi_passes_->PushRange(name);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to push range: " << status;
      }
    }
  }

  void PopRange() override {
    if (multi_passes_ != nullptr) {
      absl::Status status = multi_passes_->PopRange();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to pop range: " << status;
      }
    }
  }

  void StopPass() override {
    if (multi_passes_ != nullptr) {
      absl::Status status = multi_passes_->StopPass();
      if (!status.ok()) {
        LOG(ERROR) << "Failed to stop pass: " << status;
      }
    }
  }

  void UploadSession() override {
    auto local_xspace = std::make_unique<tensorflow::profiler::XSpace>();
    if (multi_passes_ != nullptr) {
      CHECK_OK(multi_passes_->CollectData(local_xspace.get()));
    }
    SaveXSpace(std::move(local_xspace));
  }

 protected:
  std::unique_ptr<tsl::ProfilerPasses> multi_passes_;
};

// Multi-pass profiler implementation for multi-host HLO runner execution.
// Uses KeyValueStoreInterface to synchronize pass progression across ranks.
class CollectiveMultipassHloProfiler : public MultiPassHloProfiler {
 public:
  CollectiveMultipassHloProfiler(
      absl::string_view dump_path, bool keep_xspace,
      std::shared_ptr<KeyValueStoreInterface> multipass_synchronizer,
      int task_id, int num_nodes,
      std::optional<int> collective_id = std::nullopt)
      : MultiPassHloProfiler(dump_path, keep_xspace),
        multipass_synchronizer_(std::move(multipass_synchronizer)),
        task_id_(task_id),
        num_nodes_(num_nodes),
        collective_id_(collective_id) {}

  void CreateSession() override {
    MultiPassHloProfiler::CreateSession();
    session_count_++;
    pass_idx_ = 0;
  }

  absl::StatusOr<bool> NeedMorePasses() override {
    bool local_need_more = false;
    if (multi_passes_ != nullptr) {
      ABSL_ASSIGN_OR_RETURN(local_need_more, multi_passes_->NeedMorePasses());
    }

    std::string keystr =
        collective_id_.has_value()
            ? absl::StrCat("hlo_runner_need_more_pass_", *collective_id_, "_",
                           session_count_, "_", pass_idx_)
            : absl::StrCat("hlo_runner_need_more_pass_", session_count_, "_",
                           pass_idx_);
    ABSL_ASSIGN_OR_RETURN(
        bool global_need_more,
        AllReduceOr(multipass_synchronizer_.get(), task_id_, num_nodes_,
                    local_need_more, keystr, absl::Seconds(30)));

    local_need_more_for_current_pass_ = local_need_more;
    return global_need_more;
  }

  void StartPass() override {
    if (local_need_more_for_current_pass_) {
      MultiPassHloProfiler::StartPass();
    }
  }

  void PushRange(absl::string_view name) override {
    if (local_need_more_for_current_pass_) {
      MultiPassHloProfiler::PushRange(name);
    }
  }

  void PopRange() override {
    if (local_need_more_for_current_pass_) {
      MultiPassHloProfiler::PopRange();
    }
  }

  void StopPass() override {
    if (local_need_more_for_current_pass_) {
      MultiPassHloProfiler::StopPass();
    }
    pass_idx_++;
  }

 private:
  std::shared_ptr<KeyValueStoreInterface> multipass_synchronizer_;
  int task_id_ = 0;
  int num_nodes_ = 1;
  std::optional<int> collective_id_ = std::nullopt;
  int session_count_ = 0;
  int pass_idx_ = 0;
  bool local_need_more_for_current_pass_ = true;
};

}  // namespace

absl::StatusOr<bool> AllReduceOr(KeyValueStoreInterface* kv_store, int rank_id,
                                 int num_ranks, bool local_value,
                                 absl::string_view keystr,
                                 absl::Duration timeout) {
  if (kv_store == nullptr || num_ranks <= 1) {
    return local_value;
  }
  if (rank_id < 0 || rank_id >= num_ranks) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid rank_id ", rank_id, " for num_ranks ", num_ranks));
  }

  std::string my_key = absl::StrCat(keystr, "_", rank_id);
  ABSL_RETURN_IF_ERROR(kv_store->Set(my_key, local_value ? "1" : "0"));

  bool global_or = false;
  std::string decision_key = absl::StrCat(keystr, "_decision");

  if (rank_id == 0) {
    global_or = local_value;
    const absl::Time deadline = absl::Now() + timeout;
    for (int i = 1; i < num_ranks; ++i) {
      std::string peer_key = absl::StrCat(keystr, "_", i);
      const absl::Duration remaining =
          std::max(absl::ZeroDuration(), deadline - absl::Now());
      absl::StatusOr<std::string> val_str = kv_store->Get(peer_key, remaining);
      if (!val_str.ok()) {
        // Publish an error token so peer ranks waiting on `decision_key`
        // unblock promptly rather than waiting out the full timeout.
        kv_store->Set(decision_key, kAllReduceErrorToken).IgnoreError();
        return val_str.status();
      }
      if (*val_str == "1") {
        global_or = true;
      }
    }
    ABSL_RETURN_IF_ERROR(kv_store->Set(decision_key, global_or ? "1" : "0"));
  } else {
    ABSL_ASSIGN_OR_RETURN(std::string decision_str,
                     kv_store->Get(decision_key, timeout));
    if (decision_str == kAllReduceErrorToken) {
      return absl::InternalError(
          "Rank 0 failed to gather peer values during AllReduceOr.");
    }
    global_or = (decision_str == "1");
  }

  return global_or;
}

HLORunnerProfiler::HLORunnerProfiler(absl::string_view dump_path,
                                     bool keep_xspace)
    : dump_path_(dump_path), keep_xspace_(keep_xspace) {}

HLORunnerProfiler::~HLORunnerProfiler() = default;

absl::StatusOr<std::unique_ptr<HLORunnerProfiler>> HLORunnerProfiler::Create(
    absl::string_view dump_path, bool keep_xspace,
    bool enable_multipass_profiling,
    std::shared_ptr<KeyValueStoreInterface> kv_store, int task_id,
    int num_nodes, std::optional<int> collective_id) {
  if (dump_path.empty()) {
    return absl::InvalidArgumentError(
        "Please provide a valid dump path to save XSpace results to disk.");
  }
  if (enable_multipass_profiling) {
    if (num_nodes > 1) {
      if (kv_store == nullptr) {
        return absl::InvalidArgumentError(
            "Multi-host multi-pass profiling requires a non-null "
            "KeyValueStoreInterface for collective synchronization.");
      }
      if (task_id < 0 || task_id >= num_nodes) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid task_id ", task_id, " for num_nodes ", num_nodes));
      }
      return std::make_unique<CollectiveMultipassHloProfiler>(
          dump_path, keep_xspace, std::move(kv_store), task_id, num_nodes,
          collective_id);
    }
    LOG(INFO) << "Creating Single-host MultiPassHloProfiler";
    return std::make_unique<MultiPassHloProfiler>(dump_path, keep_xspace);
  }
  return std::unique_ptr<HLORunnerProfiler>(
      new HLORunnerProfiler(dump_path, keep_xspace));
}

void HLORunnerProfiler::CreateSession() {
  auto options = tsl::ProfilerSession::DefaultOptions();
  session_ = tsl::ProfilerSession::Create(options);
}

void HLORunnerProfiler::UploadSession() {
  auto local_xspace = std::make_unique<tensorflow::profiler::XSpace>();
  if (session_ != nullptr) {
    CHECK_OK(session_->CollectData(local_xspace.get()));
  }
  SaveXSpace(std::move(local_xspace));
}

void HLORunnerProfiler::SaveXSpace(
    std::unique_ptr<tensorflow::profiler::XSpace> local_xspace) {
  xspace_ = std::move(local_xspace);
  CHECK(!dump_path_.empty());
  std::string unique_dump_path = dump_path_;
  if (session_index_ > 0) {
    absl::string_view stem = dump_path_;
    absl::string_view suffix = "";
    const std::string::size_type dot_pos = dump_path_.rfind('.');
    const std::string::size_type slash_pos = dump_path_.rfind('/');
    if (dot_pos != std::string::npos &&
        (slash_pos == std::string::npos || dot_pos > slash_pos)) {
      suffix = stem.substr(dot_pos);
      stem = stem.substr(0, dot_pos);
    }
    unique_dump_path = absl::StrCat(stem, "_", session_index_, suffix);
  }
  ++session_index_;

  LOG(INFO) << "Saving xspace result to " << unique_dump_path;
  // Save in binary format to create xprof sessions and extract device stats.
  CHECK_OK(WriteBinaryProto(tsl::Env::Default(), unique_dump_path, *xspace_));
  if (!keep_xspace_) {
    xspace_ = nullptr;
  }
}

const tensorflow::profiler::XSpace* HLORunnerProfiler::GetXSpace() {
  return xspace_.get();
}

}  // namespace xla
