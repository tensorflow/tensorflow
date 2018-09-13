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
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/collective_rma_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

RpcCollectiveExecutorMgr::RpcCollectiveExecutorMgr(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    std::unique_ptr<DeviceResolverDistributed> dev_resolver,
    std::unique_ptr<CollectiveParamResolverDistributed> param_resolver,
    WorkerCacheInterface* worker_cache, const string& task_name)
    : CollectiveExecutorMgr(config, dev_mgr, std::move(dev_resolver),
                            std::move(param_resolver)),
      worker_cache_(worker_cache),
      task_name_(task_name) {
  group_leader_ = (task_name == config.experimental().collective_group_leader())
                      ? ""
                      : config.experimental().collective_group_leader();
}

RpcCollectiveExecutorMgr::~RpcCollectiveExecutorMgr() {
  for (auto it : sequence_table_) {
    delete it.second;
  }
}

CollectiveExecutor* RpcCollectiveExecutorMgr::Create(int64 step_id) {
  CollectiveRemoteAccessDistributed* rma =
      new CollectiveRemoteAccessDistributed(dev_mgr_, dev_resolver_.get(),
                                            worker_cache_, step_id);
  return new BaseCollectiveExecutor(this, rma, step_id, dev_mgr_);
}

namespace {
// StepId must leave the most-significant 7 bits empty for future use.
static const int64 kStepIdMask = (((1uLL << 56) - 1) | (1uLL << 56));

int64 NewRandomStepId() {
  int64 step_id = random::New64();
  // Leave MS 8 bits clear for future use.
  step_id &= kStepIdMask;
  return step_id;
}
}  // namespace

void RpcCollectiveExecutorMgr::RefreshStepIdSequenceAsync(
    int64 graph_key, const StatusCallback& done) {
  if (group_leader_.empty()) {
    mutex_lock l(sequence_mu_);
    GraphKeySequence* gks = nullptr;
    auto it = sequence_table_.find(graph_key);
    if (it == sequence_table_.end()) {
      gks = new GraphKeySequence(graph_key);
      sequence_table_[graph_key] = gks;
    } else {
      gks = it->second;
    }
    gks->next_step_id_ = NewRandomStepId();
    done(Status::OK());
  } else {
    WorkerInterface* wi = worker_cache_->CreateWorker(group_leader_);
    GetStepSequenceRequest* req = new GetStepSequenceRequest;
    GetStepSequenceResponse* resp = new GetStepSequenceResponse;
    req->add_graph_key(graph_key);
    wi->GetStepSequenceAsync(
        req, resp, [this, req, resp, done](const Status& s) {
          if (!s.ok()) {
            LOG(ERROR) << "Bad response [" << s
                       << "] from GetStepSequenceAsync call to "
                       << group_leader_;
            done(s);
          } else {
            done(UpdateStepSequences(*resp));
          }
          delete req;
          delete resp;
        });
  }
}

void RpcCollectiveExecutorMgr::GetStepSequenceAsync(
    const GetStepSequenceRequest* request, GetStepSequenceResponse* response,
    const StatusCallback& done) {
  if (!group_leader_.empty()) {
    LOG(ERROR) << "GetStepSequence called at non-group-leader";
    done(errors::Internal("GetStepSequenceAsync called at non-group-leader"));
  } else {
    mutex_lock l(sequence_mu_);
    for (int64 graph_key : request->graph_key()) {
      auto it = sequence_table_.find(graph_key);
      GraphKeySequence* gks = nullptr;
      if (it == sequence_table_.end()) {
        gks = new GraphKeySequence(graph_key);
        gks->next_step_id_ = NewRandomStepId();
        sequence_table_[graph_key] = gks;
      } else {
        gks = it->second;
      }
      StepSequence* ss = response->add_step_sequence();
      ss->set_graph_key(graph_key);
      ss->set_next_step_id(gks->next_step_id_);
    }
    done(Status::OK());
  }
}

Status RpcCollectiveExecutorMgr::UpdateStepSequences(
    const GetStepSequenceResponse& resp) {
  mutex_lock l(sequence_mu_);
  for (const StepSequence& ss : resp.step_sequence()) {
    GraphKeySequence* gks = nullptr;
    auto it = sequence_table_.find(ss.graph_key());
    if (it == sequence_table_.end()) {
      gks = new GraphKeySequence(ss.graph_key());
      sequence_table_[ss.graph_key()] = gks;
    } else {
      gks = it->second;
    }
    gks->next_step_id_ = ss.next_step_id();
  }
  return Status::OK();
}

int64 RpcCollectiveExecutorMgr::NextStepId(int64 graph_key) {
  mutex_lock l(sequence_mu_);
  auto it = sequence_table_.find(graph_key);
  if (it != sequence_table_.end()) {
    return it->second->next_step_id_;
  }
  return CollectiveExecutor::kInvalidId;
}

void RpcCollectiveExecutorMgr::RetireStepId(int64 graph_key, int64 step_id) {
  mutex_lock l(sequence_mu_);
  auto it = sequence_table_.find(graph_key);
  if (it != sequence_table_.end()) {
    if (step_id == it->second->next_step_id_) {
      it->second->next_step_id_ = (it->second->next_step_id_ + 1) & kStepIdMask;
    } else {
      it->second->next_step_id_ = CollectiveExecutor::kInvalidId;
    }
  } else {
    LOG(ERROR) << "Failed to find graph_key " << graph_key << " to retire.";
  }
}

}  // namespace tensorflow
