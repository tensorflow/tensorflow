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

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system_helper.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/profiler_passes.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

// Performs an all-reduce OR across all ranks using the KeyValueStoreInterface.
// In multipass profiling, this is used to reach consensus on whether any host
// requires an additional profiling pass (`NeedMorePasses()`), ensuring all
// ranks remain synchronized and execute the same number of passes.
//
// If `num_ranks <= 1` or `kv_store == nullptr` (single rank / single host
// mode), no communication is needed and the local value is returned directly.
//
// Callers must provide a unique `keystr` for each reduction step (e.g., by
// appending the pass index) to avoid collisions with prior passes or concurrent
// reductions in the key-value store.
absl::StatusOr<bool> AllReduceOr(KeyValueStoreInterface* kv_store, int rank_id,
                                 int num_ranks, bool local_value,
                                 absl::string_view keystr,
                                 absl::Duration timeout) {
  if (kv_store == nullptr || num_ranks <= 1) {
    return local_value;
  }

  std::string my_key = absl::StrCat(keystr, "_", rank_id);
  ABSL_RETURN_IF_ERROR(kv_store->Set(my_key, local_value ? "1" : "0"));

  bool global_or = false;
  std::string decision_key = absl::StrCat(keystr, "_decision");

  if (rank_id == 0) {
    global_or = local_value;
    for (int i = 1; i < num_ranks; ++i) {
      std::string peer_key = absl::StrCat(keystr, "_", i);
      ABSL_ASSIGN_OR_RETURN(std::string val_str, kv_store->Get(peer_key, timeout));
      if (val_str == "1") {
        global_or = true;
      }
    }
    ABSL_RETURN_IF_ERROR(kv_store->Set(decision_key, global_or ? "1" : "0"));
  } else {
    ABSL_ASSIGN_OR_RETURN(std::string decision_str,
                     kv_store->Get(decision_key, timeout));
    global_or = (decision_str == "1");
  }

  return global_or;
}

// Multi-pass profiler implementation for single-host HLO runner execution.
class MultiPassHloProfiler : public HLORunnerProfiler {
 public:
  MultiPassHloProfiler(absl::string_view dump_path, bool keep_xspace)
      : HLORunnerProfiler(dump_path, keep_xspace) {}

  void CreateSession() override {
    multi_passes_ =
        tsl::ProfilerPasses::Create(tsl::ProfilerPasses::DefaultOptions());
  }

  bool NeedMorePass() override {
    return multi_passes_ != nullptr && multi_passes_->NeedMorePasses();
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
    CHECK_OK(multi_passes_->CollectData(local_xspace.get()));
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
      int task_id, int num_nodes)
      : MultiPassHloProfiler(dump_path, keep_xspace),
        multipass_synchronizer_(std::move(multipass_synchronizer)),
        task_id_(task_id),
        num_nodes_(num_nodes) {}

  bool NeedMorePass() override {
    if (multi_passes_ == nullptr) {
      return false;
    }
    bool local_need_more = multi_passes_->NeedMorePasses();

    std::string keystr = absl::StrCat("hlo_runner_need_more_pass_", pass_idx_);
    absl::StatusOr<bool> global_need_more =
        AllReduceOr(multipass_synchronizer_.get(), task_id_, num_nodes_,
                    local_need_more, keystr, absl::Seconds(30));

    if (!global_need_more.ok()) {
      LOG(ERROR) << "Failed to run collective NeedMorePass: "
                 << global_need_more.status();
      local_need_more_for_current_pass_ = local_need_more;
      return local_need_more;
    }

    local_need_more_for_current_pass_ = local_need_more;
    return *global_need_more;
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
  int pass_idx_ = 0;
  bool local_need_more_for_current_pass_ = true;
};

}  // namespace

HLORunnerProfiler::HLORunnerProfiler(absl::string_view dump_path,
                                     bool keep_xspace)
    : dump_path_(dump_path), keep_xspace_(keep_xspace) {}

HLORunnerProfiler::~HLORunnerProfiler() = default;

absl::StatusOr<std::unique_ptr<HLORunnerProfiler>> HLORunnerProfiler::Create(
    absl::string_view dump_path, bool keep_xspace,
    bool enable_multipass_profiling,
    std::shared_ptr<KeyValueStoreInterface> kv_store, int task_id,
    int num_nodes) {
  if (dump_path.empty()) {
    return absl::InvalidArgumentError(
        "Please provide a valid dump path to save XSpace results to disk.");
  }
  if (enable_multipass_profiling) {
    if (kv_store != nullptr && num_nodes > 1) {
      return std::make_unique<CollectiveMultipassHloProfiler>(
          dump_path, keep_xspace, std::move(kv_store), task_id, num_nodes);
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
  CHECK_OK(session_->CollectData(local_xspace.get()));
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
