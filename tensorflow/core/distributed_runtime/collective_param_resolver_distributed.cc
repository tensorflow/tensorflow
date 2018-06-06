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
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/protobuf/config.pb.h"

// TODO(tucker): When we're ready to enable collectives this const will
// transition to a settable config member.
static const char FLAGS_collective_group_leader[] =
    "/job:worker/replica:0/task:0";

namespace tensorflow {
namespace {
// Supports client side cancellation of WorkerInterface calls via
// registration with a CancellationManager.  Note that ParamResolverInterface
// calls are done on behalf of an Op execution which needs to abort if the
// step in which it executes is cancelled.
class CancellableCall {
 public:
  CancellableCall(CancellationManager* cancel_mgr, const string& remote_worker,
                  WorkerCacheInterface* wc)
      : cancel_mgr_(cancel_mgr), remote_worker_(remote_worker), wc_(wc) {
    wi_ = wc_->CreateWorker(remote_worker_);
  }
  virtual ~CancellableCall() { wc_->ReleaseWorker(remote_worker_, wi_); }

  virtual void IssueCall(const StatusCallback& done) = 0;

  void Start(const StatusCallback& done) {
    CancellationToken token = cancel_mgr_->get_cancellation_token();
    const bool not_yet_cancelled = cancel_mgr_->RegisterCallback(
        token, [this, token]() { opts_.StartCancel(); });
    if (not_yet_cancelled) {
      IssueCall([this, token, done](const Status& s) {
        cancel_mgr_->DeregisterCallback(token);
        done(s);
      });
    } else {
      done(errors::Cancelled("RPC Request was cancelled"));
    }
  }

 protected:
  mutable mutex mu_;
  CancellationManager* cancel_mgr_;  // Not owned
  const string remote_worker_;
  WorkerCacheInterface* wc_;  // Not owned
  WorkerInterface* wi_;       // Owned by wc_, must be released.
  CallOptions opts_;
};

class CompleteGroupCall : public CancellableCall {
 public:
  CompleteGroupCall(const CollGroupParams& group, const string& device_name,
                    CancellationManager* cancel_mgr,
                    const string& remote_worker, WorkerCacheInterface* wc)
      : CancellableCall(cancel_mgr, remote_worker, wc) {
    req_.set_group_key(group.group_key);
    req_.set_group_size(group.group_size);
    req_.set_device_type(group.device_type.type_string());
    req_.add_device_name(device_name);
  }
  ~CompleteGroupCall() override {}

  void IssueCall(const StatusCallback& done) override {
    wi_->CompleteGroupAsync(&opts_, &req_, &resp_, done);
  }

  CompleteGroupRequest req_;
  CompleteGroupResponse resp_;
};

class CompleteInstanceCall : public CancellableCall {
 public:
  CompleteInstanceCall(const CollGroupParams& group,
                       const CollInstanceParams& instance,
                       const string& node_name, const string& device_name,
                       bool is_source, CancellationManager* cancel_mgr,
                       const string& remote_worker, WorkerCacheInterface* wc)
      : CancellableCall(cancel_mgr, remote_worker, wc) {
    req_.set_name(node_name);
    req_.set_type(instance.type);
    req_.set_data_type(instance.data_type);
    instance.shape.AsProto(req_.mutable_shape());
    req_.set_group_key(group.group_key);
    req_.set_group_size(group.group_size);
    req_.set_instance_key(instance.instance_key);
    req_.set_device_type(group.device_type.type_string());
    for (int32 offset : instance.impl_details.subdiv_offsets) {
      req_.add_subdiv_offset(offset);
    }
    req_.set_device(device_name);
    req_.set_is_source(is_source);
  }

  ~CompleteInstanceCall() override {}

  void IssueCall(const StatusCallback& done) override {
    wi_->CompleteInstanceAsync(&opts_, &req_, &resp_, done);
  }

  CompleteInstanceRequest req_;
  CompleteInstanceResponse resp_;
};

}  // namespace

CollectiveParamResolverDistributed::CollectiveParamResolverDistributed(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    DeviceResolverDistributed* dev_resolver, WorkerCacheInterface* worker_cache,
    const string& task_name)
    : CollectiveParamResolverLocal(dev_mgr, dev_resolver, task_name),
      worker_cache_(worker_cache),
      group_leader_(task_name == FLAGS_collective_group_leader
                        ? ""
                        : FLAGS_collective_group_leader) {}

void CollectiveParamResolverDistributed::CompleteParamsAsync(
    const string& device, CollectiveParams* cp, CancellationManager* cancel_mgr,
    const StatusCallback& done) {
  CompleteGroupDistributed(device, cp, cancel_mgr,
                           [this, device, cp, cancel_mgr, done](
                               const Status& s, const GroupRec* gr) {
                             if (s.ok()) {
                               CompleteInstanceDistributed(device, gr, cp,
                                                           cancel_mgr, done);
                             } else {
                               done(s);
                             }
                           });
}

void CollectiveParamResolverDistributed::CompleteGroupAsync(
    const CompleteGroupRequest* request, CompleteGroupResponse* response,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  CollectiveParams cp;
  cp.group.group_key = request->group_key();
  cp.group.group_size = request->group_size();
  cp.group.device_type = DeviceType(request->device_type());
  for (const string& dn : request->device_name()) {
    cp.instance.device_names.push_back(dn);
  }
  CompleteGroupDistributed(
      cp.instance.device_names[0], &cp, cancel_mgr,
      [this, response, done](const Status& s, const GroupRec* gr) {
        if (s.ok()) {
          mutex_lock l(gr->mu);
          response->set_group_key(gr->group.group_key);
          response->set_group_size(gr->group.group_size);
          response->set_device_type(gr->group.device_type.type_string());
          response->set_num_tasks(gr->task_set.size());
          for (const string& dn : gr->device_list) {
            response->add_device_name(dn);
          }
          for (const string& tn : gr->task_list) {
            response->add_task_name(tn);
          }
        } else {
          LOG(ERROR) << "Bad status from CompleteGroupDistributed: " << s;
        }
        done(s);
      });
}

void CollectiveParamResolverDistributed::CompleteInstanceAsync(
    const CompleteInstanceRequest* request, CompleteInstanceResponse* response,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  CollectiveParams* cp = new CollectiveParams;
  cp->name = request->name();
  cp->group.group_key = request->group_key();
  cp->group.group_size = request->group_size();
  cp->group.device_type = DeviceType(request->device_type());
  cp->instance.type = CollectiveType(request->type());
  cp->instance.instance_key = request->instance_key();
  cp->instance.data_type = request->data_type();
  cp->instance.shape = TensorShape(request->shape());
  for (int32 offset : request->subdiv_offset()) {
    cp->instance.impl_details.subdiv_offsets.push_back(offset);
  }
  VLOG(1) << "New cp " << cp << " for device " << request->device() << " : "
          << cp->ToString();
  StatusCallback done_and_cleanup = [this, cp, done](const Status& s) {
    done(s);
    delete cp;
  };
  // Start by completing the group.
  CompleteGroupDistributed(
      request->device(), cp, cancel_mgr,
      [this, cp, request, response, cancel_mgr, done_and_cleanup](
          const Status& cg_status, const GroupRec* gr) {
        if (cg_status.ok()) {
          // Then complete the instance.
          CompleteInstanceDistributed(
              request->device(), gr, cp, cancel_mgr,
              [this, gr, cp, response,
               done_and_cleanup](const Status& ci_status) {
                if (ci_status.ok()) {
                  // Now source_rank should be known, so
                  // retrieve it.
                  FindInstanceRec(
                      gr, cp,
                      [this, gr, cp, response, done_and_cleanup](
                          const Status& fi_status, InstanceRec* ir) {
                        if (fi_status.ok()) {
                          mutex_lock l(ir->out_mu);
                          response->set_instance_key(cp->instance.instance_key);
                          response->set_source_rank(ir->source_rank);
                          done_and_cleanup(fi_status);
                        } else {
                          done_and_cleanup(fi_status);
                        }
                      });
                } else {
                  done_and_cleanup(ci_status);
                }
              });
        } else {
          done_and_cleanup(cg_status);
        }
      });
}

bool CollectiveParamResolverDistributed::GroupIsCached(int32 group_key) {
  mutex_lock l(group_mu_);
  const auto& it = group_table_.find(group_key);
  return it != group_table_.end();
}

Status CollectiveParamResolverDistributed::UpdateGroupCache(
    const CompleteGroupResponse& resp) {
  // Build a new record from resp.
  std::unique_ptr<GroupRec> gr(new GroupRec);
  mutex_lock grl(gr->mu);
  gr->group.device_type = DeviceType(resp.device_type());
  gr->group.group_key = resp.group_key();
  gr->group.group_size = resp.group_size();
  gr->group.num_tasks = resp.num_tasks();
  if (resp.device_name_size() != gr->group.group_size) {
    return errors::Internal(
        "CompleteGroupResponse group_size doesn't match device_name list");
  }
  for (const string& dn : resp.device_name()) {
    gr->device_set.insert(dn);
    gr->device_list.push_back(dn);
  }
  if (resp.task_name_size() != gr->group.group_size) {
    return errors::Internal(
        "CompleteGroupResponse group_size doesn't match task_name list");
  }
  for (const string& tn : resp.task_name()) {
    gr->task_list.push_back(tn);
    gr->task_set.insert(tn);
  }
  CHECK_EQ(gr->task_set.size(), gr->group.num_tasks);
  {
    // Group membership should never change. Once a record is in group_table_
    // it never gets removed.
    mutex_lock l(group_mu_);
    auto it = group_table_.find(gr->group.group_key);
    if (it == group_table_.end()) {
      group_table_[gr->group.group_key] = std::move(gr);
    }
  }
  return Status::OK();
}

void CollectiveParamResolverDistributed::CompleteGroupDistributed(
    const string& device, CollectiveParams* cp, CancellationManager* cancel_mgr,
    const GroupRecCallback& done) {
  VLOG(1) << "CompleteGroupDistributed group_key=" << cp->group.group_key
          << " dev: " << device << " is_leader=" << (group_leader_.empty());
  if (group_leader_.empty()) {
    // This is the group leader, so resolution is local.
    return CompleteGroupLocal(device, cp, done);
  } else if (!GroupIsCached(cp->group.group_key)) {
    // Need to update Group cache from the leader.
    CompleteGroupCall* call = new CompleteGroupCall(
        cp->group, device, cancel_mgr, group_leader_, worker_cache_);
    call->Start([this, device, cp, call, done](const Status& s) {
      if (s.ok()) {
        Status status = UpdateGroupCache(call->resp_);
        if (status.ok()) {
          CompleteGroupLocal(device, cp, done);
        } else {
          done(status, nullptr);
        }
      } else {
        done(s, nullptr);
      }
      delete call;
    });
    return;
  } else {
    return CompleteGroupLocal(device, cp, done);
  }
}

bool CollectiveParamResolverDistributed::InstanceIsCached(int32 instance_key) {
  mutex_lock l(instance_mu_);
  const auto& it = instance_table_.find(instance_key);
  return it != instance_table_.end();
}

void CollectiveParamResolverDistributed::UpdateInstanceCache(
    const GroupRec* gr, CollectiveParams* cp,
    const CompleteInstanceResponse& resp, const StatusCallback& done) {
  Notification note;
  InstanceRec* ir = nullptr;
  int32 source_rank = resp.source_rank();

  auto continue_with_ir = [this, cp, &ir, source_rank, done](const Status& s) {
    if (!s.ok()) {
      done(s);
      return;
    }
    Status status;
    do {
      mutex_lock l(ir->out_mu);
      if (ir->source_rank != source_rank) {
        if (ir->source_rank >= 0) {
          ir->status = errors::Internal(
              "UpdateInstanceCache: CompleteInstanceResponse for instance ",
              cp->instance.instance_key, " gives source_rank=", source_rank,
              " but cache already holds value=", ir->source_rank);
          status = ir->status;
          break;
        }
        ir->source_rank = source_rank;
      }
      if (ir->known_count < cp->group.group_size) {
        ir->known_count = cp->group.group_size;
        if (ir->known.size() != cp->group.group_size) {
          ir->status = errors::Internal(
              "UpdateInstanceCache:: CompleteInstanceResponse for instance ",
              cp->instance.instance_key, " has known.size()=", ir->known.size(),
              " < group_size=", cp->group.group_size);
          status = ir->status;
          break;
        }
        for (int i = 0; i < ir->known.size(); ++i) {
          ir->known[i] = true;
        }
      }
      status = ir->status;
    } while (false);
    // Callback outside of lock.
    done(status);
  };

  FindInstanceRec(
      gr, cp, [this, &ir, continue_with_ir](const Status s, InstanceRec* irec) {
        ir = irec;
        continue_with_ir(s);
      });
}

void CollectiveParamResolverDistributed::CompleteInstanceDistributed(
    const string& device, const GroupRec* gr, CollectiveParams* cp,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  if (group_leader_.empty()) {
    // This is the group leader so resolution is local.
    return CompleteInstanceLocal(device, gr, cp, cp->is_source, done);
  } else if (InstanceIsCached(cp->instance.instance_key)) {
    return CompleteInstanceLocal(device, gr, cp, cp->is_source, done);
  } else {
    CompleteInstanceCall* call = new CompleteInstanceCall(
        cp->group, cp->instance, cp->name, device, cp->is_source, cancel_mgr,
        group_leader_, worker_cache_);
    call->Start([this, device, gr, cp, call, done](const Status& s) {
      if (s.ok()) {
        UpdateInstanceCache(
            gr, cp, call->resp_, [this, device, gr, cp, done](const Status& s) {
              if (!s.ok()) {
                done(s);
              } else {
                CompleteInstanceLocal(device, gr, cp, cp->is_source, done);
              }
            });
      } else {
        done(s);
      }
      delete call;
    });
    return;
  }
}

}  // namespace tensorflow
