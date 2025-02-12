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

#include "absl/strings/escaping.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/cancellable_call.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

class CompleteGroupCall : public CancellableCall {
 public:
  CompleteGroupCall(const CollGroupParams& group,
                    const DeviceAttributes& device,
                    CancellationManager* cancel_mgr,
                    const string& remote_worker, WorkerCacheInterface* wc)
      : CancellableCall(cancel_mgr, remote_worker, wc) {
    req_.set_group_key(group.group_key);
    req_.set_group_size(group.group_size);
    req_.set_device_type(group.device_type.type_string());
    *req_.mutable_device_attributes() = device;
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
    req_.set_step_id(instance.step_id);
    req_.set_data_type(instance.data_type);
    instance.shape.AsProto(req_.mutable_shape());
    req_.set_group_key(group.group_key);
    req_.set_group_size(group.group_size);
    req_.set_instance_key(instance.instance_key);
    req_.set_device_type(group.device_type.type_string());
    for (int32_t offset : instance.impl_details.subdiv_offsets) {
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
    DeviceResolverDistributed* dev_resolver,
    NcclCommunicatorInterface* nccl_communicator,
    WorkerCacheInterface* worker_cache, const string& task_name)
    : CollectiveParamResolverLocal(config, dev_mgr, dev_resolver,
                                   nccl_communicator, task_name),
      worker_cache_(worker_cache),
      group_leader_(task_name == config.experimental().collective_group_leader()
                        ? ""
                        : config.experimental().collective_group_leader()) {
  VLOG(1) << "CompleteParamResolverDistributed ctor task={" << task_name
          << "} config.collective_group_leader={"
          << config.experimental().collective_group_leader() << "}"
          << " config.collective_nccl={"
          << config.experimental().collective_nccl() << "}";
}

void CollectiveParamResolverDistributed::CompleteParamsAsync(
    const DeviceAttributes& device, CollectiveParams* cp,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  VLOG(1) << "CompleteParams distributed " << device.name() << " for " << cp
          << ": " << cp->ToString();
  if (cp->run_group_initialization) {
    CompleteGroupDistributed(
        device, &cp->group, cancel_mgr,
        [this, device, cp, cancel_mgr, done](absl::Status s) {
          if (s.ok()) {
            std::vector<DeviceAttributes> devices;
            devices.reserve(cp->group.group_size);
            for (const CollGroupMember& m : cp->group.members) {
              devices.push_back(m.device);
            }
            s = dev_resolver_->UpdateDeviceAttributes(devices);
          }
          if (s.ok()) {
            CompleteInstanceDistributed(device.name(), cp, cancel_mgr, done);
          } else {
            done(s);
          }
        });
  } else {
    // For Collective V3 ops, group is already initialized. Fetch attributes
    // for the already initialized group to pass to Insitance initialization.
    auto s = LookupGroup(cp->group.group_key, &cp->group);
    if (s.ok()) {
      CompleteInstanceDistributed(device.name(), cp, cancel_mgr, done);
    } else {
      done(s);
    }
  }
}

void CollectiveParamResolverDistributed::CompleteGroupAsync(
    const DeviceAttributes& device, CollGroupParams* group_params,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  CompleteGroupDistributed(device, group_params, cancel_mgr, done);
}

void CollectiveParamResolverDistributed::CompleteInstanceAsync(
    const CompleteInstanceRequest* request, CompleteInstanceResponse* response,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  GroupRec* gr = GetCachedGroup(request->group_key());
  if (gr == nullptr) {
    done(errors::FailedPrecondition(
        "group ", request->group_key(),
        " not found. This normally means the server has restarted"));
    return;
  }
  CollectiveParams* cp = new CollectiveParams;
  {
    mutex_lock l(gr->mu);
    if (!gr->status.ok()) {
      done(gr->status);
      return;
    } else if (gr->group.members.size() != gr->group.group_size) {
      done(errors::FailedPrecondition(
          "group ", request->group_key(),
          " failed to resolve. This normally means the server has restarted"));
      return;
    }
    cp->group = gr->group;
  }
  cp->name = request->name();
  cp->instance.type = CollectiveType(request->type());
  cp->instance.instance_key = request->instance_key();
  cp->instance.step_id = request->step_id();
  cp->instance.data_type = request->data_type();
  cp->instance.shape = TensorShape(request->shape());
  cp->is_source = request->is_source();
  for (int32_t offset : request->subdiv_offset()) {
    cp->instance.impl_details.subdiv_offsets.push_back(offset);
  }
  StatusCallback done_and_cleanup = [cp, done](const absl::Status& s) {
    done(s);
    cp->Unref();
  };
  CompleteInstanceDistributed(
      request->device(), cp, cancel_mgr,
      [this, cp, response, done_and_cleanup](absl::Status status) {
        if (status.ok()) {
          // Now source_rank should be known, so retrieve it.
          bool created_irec;
          InstanceRec* ir = GetOrCreateInstanceRec(cp, &created_irec);
          {
            mutex_lock l(ir->mu);
            status = ir->status;
            if (ir->status.ok()) {
              response->set_instance_key(cp->instance.instance_key);
              response->set_source_rank(ir->source_rank);
            }
          }
        }
        done_and_cleanup(status);
      });
}

CollectiveParamResolverDistributed::GroupRec*
CollectiveParamResolverDistributed::GetCachedGroup(int32_t group_key) {
  mutex_lock l(group_mu_);
  auto it = group_table_.find(group_key);
  if (it == group_table_.end()) {
    return nullptr;
  }
  return it->second.get();
}

absl::Status CollectiveParamResolverDistributed::UpdateGroupCache(
    const CompleteGroupResponse& resp) {
  // Build a new record from resp.
  std::unique_ptr<GroupRec> gr(new GroupRec);
  {
    mutex_lock grl(gr->mu);
    gr->group.device_type = DeviceType(resp.device_type());
    gr->group.group_key = resp.group_key();
    gr->group.group_size = resp.group_size();
    gr->group.num_tasks = resp.num_tasks();
    if (resp.device_attributes().empty()) {
      return errors::Internal(
          "CompleteGroupResponse device_attributes is empty. Make sure you're "
          "running the same version of Tensorflow on all workers.");
    }
    if (resp.device_attributes_size() != gr->group.group_size) {
      return errors::Internal(
          "CompleteGroupResponse group_size doesn't match device_name list");
    }
    gr->group.members.reserve(resp.device_attributes().size());
    for (const DeviceAttributes& device : resp.device_attributes()) {
      CollGroupMember member;
      member.device = device;
      gr->group.members.push_back(std::move(member));
      gr->incarnations_by_device_name[device.name()] = device.incarnation();
    }
    gr->group.runtime_details.communicator_key = resp.communicator_key();
    FinishGroup(gr.get());
  }
  GroupRec* previous_gr = nullptr;
  {
    // Group membership should never change. Once a record is in group_table_
    // it never gets removed.
    mutex_lock l(group_mu_);
    auto it = group_table_.find(resp.group_key());
    if (it == group_table_.end()) {
      VLOG(2) << "UpdateGroupCache: communicator_key="
              << absl::CEscape(resp.communicator_key());
      group_table_[gr->group.group_key] = std::move(gr);
    } else {
      previous_gr = it->second.get();
    }
  }
  if (previous_gr != nullptr) {
    mutex_lock grl(previous_gr->mu);
    if (previous_gr->group.runtime_details.communicator_key !=
        resp.communicator_key()) {
      return errors::Internal(
          "UpdateGroupCache: CompleteGroupResponse for group ",
          resp.group_key(),
          " gives communicator_key=", absl::CEscape(resp.communicator_key()),
          " but cache already holds communicator_key=",
          absl::CEscape(previous_gr->group.runtime_details.communicator_key));
    }
  }
  return absl::OkStatus();
}

void CollectiveParamResolverDistributed::CompleteGroupDistributed(
    const DeviceAttributes& device, CollGroupParams* group_params,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  VLOG(1) << "CompleteGroupDistributed group_key=" << group_params->group_key
          << " dev: " << device.name()
          << " is_leader=" << (group_leader_.empty());
  if (group_leader_.empty()) {
    // This is the group leader, so resolution is local.
    return CompleteGroupLocal(device, group_params, cancel_mgr, done);
  } else if (GetCachedGroup(group_params->group_key) == nullptr) {
    // Need to update Group cache from the leader.
    CompleteGroupCall* call = new CompleteGroupCall(
        *group_params, device, cancel_mgr, group_leader_, worker_cache_);
    CancellationToken abortion_token =
        abortion_cancel_mgr_.get_cancellation_token();
    bool already_aborted = !abortion_cancel_mgr_.RegisterCallback(
        abortion_token, [call] { call->Cancel(); });
    if (already_aborted) {
      done(errors::Cancelled("collective ops already aborted"));
      delete call;
      return;
    }
    call->Start([this, device, group_params, call, cancel_mgr, abortion_token,
                 done](const absl::Status& s) {
      abortion_cancel_mgr_.DeregisterCallback(abortion_token);
      if (s.ok()) {
        absl::Status status = UpdateGroupCache(call->resp_);
        if (status.ok()) {
          CompleteGroupLocal(device, group_params, cancel_mgr, done);
        } else {
          done(status);
        }
      } else {
        done(s);
      }
      delete call;
    });
    return;
  } else {
    return CompleteGroupLocal(device, group_params, cancel_mgr, done);
  }
}

bool CollectiveParamResolverDistributed::InstanceIsCached(
    int32_t group_key, const CollInstanceParams& instance) {
  mutex_lock l(instance_mu_);
  auto group_it = instance_table_.find(group_key);
  if (group_it == instance_table_.end()) {
    return false;
  }
  auto instance_it =
      group_it->second.find({instance.step_id, instance.instance_key});
  return instance_it != group_it->second.end();
}

absl::Status CollectiveParamResolverDistributed::UpdateInstanceCache(
    CollectiveParams* cp, const CompleteInstanceResponse& resp) {
  int32_t source_rank = resp.source_rank();
  bool created_irec;
  InstanceRec* ir = GetOrCreateInstanceRec(cp, &created_irec);
  mutex_lock l(ir->mu);
  if (!ir->status.ok()) {
    return ir->status;
  }
  if (ir->source_rank != source_rank) {
    if (ir->source_rank >= 0) {
      ir->status = errors::Internal(
          "UpdateInstanceCache: CompleteInstanceResponse for instance ",
          cp->instance.instance_key, " gives source_rank=", source_rank,
          " but cache already holds value=", ir->source_rank);
      return ir->status;
    }
    ir->source_rank = source_rank;
  }
  if (ir->known_count < cp->group.group_size) {
    ir->known_count = cp->group.group_size;
    const int ir_known_size = ir->known.size();
    if (ir_known_size != cp->group.group_size) {
      ir->status = errors::Internal(
          "UpdateInstanceCache:: CompleteInstanceResponse for instance ",
          cp->instance.instance_key, " has known.size()=", ir->known.size(),
          " < group_size=", cp->group.group_size);
      return ir->status;
    }
    for (int i = 0; i < ir_known_size; ++i) {
      ir->known[i] = true;
    }
  }
  return ir->status;
}

void CollectiveParamResolverDistributed::CompleteInstanceDistributed(
    const string& device, CollectiveParams* cp, CancellationManager* cancel_mgr,
    const StatusCallback& done) {
  if (group_leader_.empty()) {
    // This is the group leader so resolution is local.
    return CompleteInstanceLocal(device, cp, done);
  } else if (InstanceIsCached(cp->group.group_key, cp->instance)) {
    return CompleteInstanceLocal(device, cp, done);
  } else {
    CompleteInstanceCall* call = new CompleteInstanceCall(
        cp->group, cp->instance, cp->name, device, cp->is_source, cancel_mgr,
        group_leader_, worker_cache_);
    CancellationToken abortion_token =
        abortion_cancel_mgr_.get_cancellation_token();
    bool already_aborted = !abortion_cancel_mgr_.RegisterCallback(
        abortion_token, [call] { call->Cancel(); });
    if (already_aborted) {
      done(errors::Cancelled("collective ops already aborted"));
      delete call;
      return;
    }
    call->Start([this, device, cp, call, abortion_token, done](absl::Status s) {
      abortion_cancel_mgr_.DeregisterCallback(abortion_token);
      if (s.ok()) {
        s = UpdateInstanceCache(cp, call->resp_);
      }
      if (s.ok()) {
        CompleteInstanceLocal(device, cp, done);
      } else {
        done(s);
      }
      delete call;
    });
    return;
  }
}

void CollectiveParamResolverDistributed::StartAbort(const absl::Status& s) {
  {
    mutex_lock l(status_mu_);
    if (!status_.ok()) {
      VLOG(2) << "CollectiveParamResolverDistributed already aborted. Ignoring "
                 "subsequent abortion with status: "
              << s;
      return;
    }
    status_ = s;
  }
  StartAbortLocal(s);
  abortion_cancel_mgr_.StartCancel();
}

}  // namespace tensorflow
