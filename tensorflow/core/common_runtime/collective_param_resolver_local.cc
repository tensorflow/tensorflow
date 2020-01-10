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
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"

#include <stddef.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

void CollectiveParamResolverLocal::InstanceRec::WaitForOutMu(mutex_lock& lock) {
  while (!out_mu_available) out_cv.wait(lock);
}

CollectiveParamResolverLocal::CollectiveParamResolverLocal(
    const ConfigProto& config, const DeviceMgr* dev_mgr,
    DeviceResolverInterface* dev_resolver, const string& task_name)
    : nccl_(config.experimental().collective_nccl()),
      dev_mgr_(dev_mgr),
      dev_resolver_(dev_resolver),
      task_name_(task_name) {}

void CollectiveParamResolverLocal::CompleteGroupAsync(
    const CompleteGroupRequest* request, CompleteGroupResponse* response,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  done(
      errors::Internal("CompleteGroup is not implemented by "
                       "CollectiveParamResolverLocal which is "
                       "intended only for non-distributed deployment."));
}

namespace {
const char* GetCollectiveName(const CollectiveParams* cp, bool nccl) {
  switch (cp->instance.type) {
    case BROADCAST_COLLECTIVE:
      return "HierarchicalTreeBroadcast";

    case REDUCTION_COLLECTIVE:
      return nccl ? "NcclReduce" : "RingReduce";

    case GATHER_COLLECTIVE:
      return "RingGather";

    default:
      return "undef";
  }
}
}  // namespace

void CollectiveParamResolverLocal::CompleteGroupLocal(
    const string& device, CollectiveParams* cp, const GroupRecCallback& done) {
  VLOG(1) << "CompleteGroupLocal device=" << device << " cp: " << cp << ": "
          << cp->ToString();
  std::vector<StatusCallback> to_be_called;
  GroupRec* gr = nullptr;
  Status status;
  {
    mutex_lock l(group_mu_);
    auto it = group_table_.find(cp->group.group_key);
    if (it == group_table_.end()) {
      gr = new GroupRec;
      gr->group.group_key = cp->group.group_key;
      gr->group.group_size = cp->group.group_size;
      gr->group.device_type = cp->group.device_type;

      // Initialize group runtime details.
      CollectiveImplementationInterface* col_impl;
      // Try to lookup a NCCL collective kernel.  This will return error status
      // if `NcclReduce` kernel is not present in the registry, e.g. on an
      // environment that does not support NCCL.
      status = CollectiveRegistry::LookupParamResolverInstance("NcclReduce",
                                                               &col_impl);
      if (!status.ok()) {
        // Fallback to non-NCCL collective.
        status = CollectiveRegistry::LookupParamResolverInstance(
            GetCollectiveName(cp, /*nccl=*/false), &col_impl);
      }
      if (status.ok()) {
        status = col_impl->InitializeCollectiveGroupRuntimeDetails(
            &gr->group.runtime_details);
      }

      if (!status.ok()) {
        done(status, gr);
        return;
      }

      // Store GroupRec in group_table_ which is shared between all devices on
      // this worker.
      group_table_[gr->group.group_key].reset(gr);
      VLOG(2) << "New group_key=" << gr->group.group_key
              << " group_size=" << gr->group.group_size
              << " runtime_details=" << gr->group.runtime_details.ToString();
    } else {
      gr = it->second.get();
    }
  }
  {
    mutex_lock gr_lock(gr->mu);
    // If there is ever an error associated with a group key, we store the error
    // status and invoke all waiting and future callbacks with this error
    // status.
    VLOG(2) << "gr device_type=" << gr->group.device_type
            << " cp device_type=" << cp->group.device_type
            << " current device=" << device;
    if (gr->status.ok()) {
      // Check for consistency with existing GroupRec.
      if (cp->group.device_type != gr->group.device_type) {
        gr->status = errors::Internal(
            "Collective Op ", cp->name, " is assigned to device ", device,
            " with type ", cp->group.device_type.type_string(),
            " and group_key ", cp->group.group_key, " but that group has type ",
            gr->group.device_type.type_string());
      } else if (cp->group.group_size != gr->group.group_size) {
        gr->status = errors::Internal(
            "Collective Op ", cp->name, " has group_size ",
            cp->group.group_size, " and group_key ", cp->group.group_key,
            " but that group has size ", gr->group.group_size);
      }
    }
    if (gr->status.ok()) {
      // Insert device if not already present.
      auto it = gr->device_set.find(device);
      if (it == gr->device_set.end()) {
        if (gr->device_set.size() == gr->group.group_size) {
          // The group is already full.
          gr->status = errors::Internal(
              "Collective Op ", cp->name, " is assigned to device ", device,
              " and group_key ", cp->group.group_key,
              " but that group doesn't contain that device.");
        } else {
          // This is a new device that has not yet joined the group.
          gr->device_set.insert(device);
          gr->device_list.push_back(device);
          DeviceNameUtils::ParsedName parsed_device;
          DeviceNameUtils::ParseFullName(device, &parsed_device);
          string task_name = strings::StrCat("/job:", parsed_device.job,
                                             "/replica:", parsed_device.replica,
                                             "/task:", parsed_device.task);
          gr->task_set.insert(task_name);
          gr->task_list.push_back(task_name);
          gr->group.num_tasks = static_cast<int32>(gr->task_set.size());
          if (VLOG_IS_ON(1)) {
            string dev_buf;
            for (const auto& d : gr->device_set) {
              strings::StrAppend(&dev_buf, ",", d);
            }
            VLOG(1) << "CompleteGroupLocal group_key=" << gr->group.group_key
                    << " group_size=" << gr->group.group_size << " (current"
                    << " devices)=(" << dev_buf << ") (number of"
                    << " devices pending)="
                    << (gr->group.group_size - gr->device_set.size());
          }
        }
      }
    }

    if (gr->status.ok()) {
      cp->group.runtime_details = gr->group.runtime_details;
      // If the group is not yet complete, queue to wait for it.
      VLOG(2) << "group_size " << gr->group.group_size << " set size "
              << gr->device_set.size() << " gr " << gr;

      if (gr->device_set.size() < gr->group.group_size) {
        gr->waiting.push_back(std::bind(done, std::placeholders::_1, gr));
        return;
      }
      CHECK_EQ(gr->device_set.size(), gr->group.group_size);
    }
    // At this point, we either have a full group, or an error status.  Ensure
    // that all callbacks are invoked with the appropriate status.
    if (!gr->waiting.empty()) {
      std::swap(to_be_called, gr->waiting);
    }
    status = gr->status;
  }
  done(status, gr);
  for (int i = 0; i < to_be_called.size(); ++i) {
    to_be_called[i](status);
  }
}

namespace {
struct DevRec {
  string task;
  string device;
  int original_rank;
  int local_rank;
  int global_rank;
  const DeviceLocality* locality;
};
typedef std::unordered_map<string, DevRec> TaskDeviceMap;
typedef std::unordered_map<string, TaskDeviceMap> GlobalDeviceMap;

// Create a populated GlobalDeviceMap from CollInstanceParams and localities.
GlobalDeviceMap BuildDevRecs(const CollInstanceParams& ip,
                             const std::vector<DeviceAttributes>& attributes) {
  GlobalDeviceMap gdm;
  CHECK_EQ(ip.device_names.size(), ip.task_names.size());
  CHECK_EQ(ip.device_names.size(), attributes.size());
  for (int i = 0; i < ip.device_names.size(); ++i) {
    TaskDeviceMap& tdm = gdm[ip.task_names[i]];
    DevRec* dr = &tdm[ip.device_names[i]];
    dr->task = ip.task_names[i];
    dr->device = ip.device_names[i];
    dr->original_rank = i;
    dr->local_rank = 0;   // Will be populated later by OrderTaskDeviceMap.
    dr->global_rank = 0;  // Will be populated later by EstablishGlobalRank.
    dr->locality = &attributes[i].locality();
  }
  return gdm;
}

bool ParseRingOrder(const string& gpu_ring_order_str, TaskDeviceMap* tdm) {
  std::vector<string> split_gpu_ring_order_str =
      str_util::Split(gpu_ring_order_str, ',');
  if (split_gpu_ring_order_str.size() != tdm->size()) return false;

  // gpu id -> local rank
  gtl::FlatMap<int32, int32> gpu_ranks;
  for (int32 rank = 0;
       rank < static_cast<int32>(split_gpu_ring_order_str.size()); ++rank) {
    int32 tmp;
    if (strings::safe_strto32(split_gpu_ring_order_str[rank], &tmp)) {
      gpu_ranks[tmp] = rank;
    } else {
      return false;
    }
  }

  for (auto& tdm_it : *tdm) {
    DeviceNameUtils::ParsedName parsed_name;
    DevRec* dr = &tdm_it.second;
    if (!DeviceNameUtils::ParseFullName(dr->device, &parsed_name)) {
      return false;
    }
    auto rank_it = gpu_ranks.find(parsed_name.id);
    if (rank_it == gpu_ranks.end()) return false;
    dr->local_rank = rank_it->second;
  }
  VLOG(2) << "Assigned local ranks based on ring order " << gpu_ring_order_str;
  return true;
}

void OrderTaskDeviceMap(const string& gpu_ring_order, TaskDeviceMap* tdm) {
  CHECK_GT(tdm->size(), 0);  // Should never be called with 0 devices

  // If a valid ring order has been passed in via ConfigProto, use that.
  if (ParseRingOrder(gpu_ring_order, tdm)) return;

  // Either no ring order was passed in, or the format was unexpected.
  // We now assign a ring order based on link strengths.  Note that this
  // algorithm is not optimal and may not always find the best ring order.
  int least_rank = -1;
  string next_device;
  std::set<string> selected;
  // Starting device is one with the least initial rank.
  for (const auto& it : *tdm) {
    if (least_rank < 0 || it.second.original_rank < least_rank) {
      least_rank = it.second.original_rank;
      next_device = it.second.device;
    }
  }
  CHECK_GE(least_rank, 0);
  DeviceNameUtils::ParsedName parsed_name;
  CHECK(DeviceNameUtils::ParseFullName(next_device, &parsed_name));
  // NOTE: InterconnectLink has only a device_id, nothing more, so for
  // the time being if there's more than one device at a task we
  // assume they're all GPUs.

  int next_rank = 0;
  while (true) {
    selected.insert(next_device);
    auto next_dev_it = tdm->find(next_device);
    CHECK(next_dev_it != tdm->end());
    DevRec* dr = &next_dev_it->second;
    dr->local_rank = next_rank;
    ++next_rank;
    if (selected.size() == tdm->size()) {
      break;
    }
    // For the present time we assume Locality links only cover GPUs.
    // For multiple CPUs, just take them in order.
    const InterconnectLink* best_link = nullptr;
    if (parsed_name.type == "GPU") {
      for (const InterconnectLink& il : dr->locality->links().link()) {
        parsed_name.id = il.device_id();
        string endpoint_device =
            DeviceNameUtils::ParsedNameToString(parsed_name);
        // Skip the device if we've already seen it.
        if (selected.find(endpoint_device) != selected.end()) {
          continue;
        }
        // Skip the device if it is not participating in this collective
        // instance.
        if (tdm->find(endpoint_device) == tdm->end()) {
          continue;
        }
        if (best_link == nullptr || il.strength() > best_link->strength()) {
          best_link = &il;
        }
      }
    }
    if (best_link != nullptr) {
      // Follow the best edge
      parsed_name.id = best_link->device_id();
      next_device = DeviceNameUtils::ParsedNameToString(parsed_name);
    } else {
      // No good edges, alas. Pick the lowest initial rank among remaining
      // devices.
      least_rank = -1;
      for (const auto& it : *tdm) {
        if (selected.find(it.second.device) != selected.end()) {
          continue;
        }
        if (least_rank < 0 || it.second.original_rank < least_rank) {
          least_rank = it.second.original_rank;
          next_device = it.second.device;
        }
      }
      CHECK_GE(least_rank, 0);
    }
  }
}

// The first time a shared CollectiveParams is established for a
// shared set of instances we compute a good rank order for all the
// devices in the group, that is appropriate for a ring algorithm.
// This order need not be the same across different instance groups
// sharing the same device group where there is more than one good
// order.
GlobalDeviceMap EstablishGlobalRank(
    CollectiveParams* cp, const std::vector<DeviceAttributes>& attributes) {
  VLOG(1) << "EstablishGlobalRank";
  GlobalDeviceMap gdm = BuildDevRecs(cp->instance, attributes);
  for (auto& iter : gdm) {
    TaskDeviceMap& tdm = iter.second;
    OrderTaskDeviceMap(cp->instance.gpu_ring_order, &tdm);
  }
  // Connect the global rank order by the order in which tasks first appear.
  std::set<string> ordered_tasks;
  int next_rank = 0;
  for (int i = 0; i < cp->instance.task_names.size(); ++i) {
    const string& task_name = cp->instance.task_names[i];
    if (ordered_tasks.find(task_name) != ordered_tasks.end()) {
      continue;
    }
    ordered_tasks.insert(task_name);
    TaskDeviceMap* tdm = &gdm[task_name];
    for (auto& it : *tdm) {
      it.second.global_rank = it.second.local_rank + next_rank;
    }
    next_rank += tdm->size();
  }
  return gdm;
}

// Count the devices associated with each task and set
// cp->same_num_devices_per_task.  Requires cp->instance.task_names
// be sorted.
void SetDevPerTask(CollectiveParams* cp) {
  cp->instance.num_devices_per_task.clear();
  const string* last_task_name = &cp->instance.task_names[0];
  int count = 0;
  for (const string& task_name : cp->instance.task_names) {
    if (task_name == *last_task_name) {
      ++count;
    } else {
      cp->instance.num_devices_per_task[*last_task_name] = count;
      count = 1;
      last_task_name = &task_name;
    }
  }
  cp->instance.num_devices_per_task[*last_task_name] = count;

  cp->instance.same_num_devices_per_task = false;
  int dev_per_task = -1;
  for (const auto& task_dev : cp->instance.num_devices_per_task) {
    if (dev_per_task == -1) {
      dev_per_task = task_dev.second;
    } else if (dev_per_task != task_dev.second) {
      return;
    }
  }
  cp->instance.same_num_devices_per_task = true;
  CHECK_EQ((cp->group.group_size % cp->group.num_tasks), 0);
}

// Sort cp->instance.device_names lexicographically, but do by first
// computing a reordering permutation so we can keep cp->instance.task_names
// in corresponding order.
void SortDevicesAndTasks(CollectiveParams* cp) {
  VLOG(1) << "SortDevicesAndTasks " << cp << " instance " << &cp->instance;
  CHECK(cp);
  CHECK_EQ(cp->group.group_size, cp->instance.device_names.size());
  CHECK_EQ(cp->group.group_size, cp->instance.task_names.size());
  std::vector<int> perm(cp->group.group_size);
  // TODO(tucker): substitute std::iota when the windows build supports it.
  // std::iota(perm.begin(), perm.end(), 0);
  for (int i = 0; i < perm.size(); ++i) {
    perm[i] = i;
  }
  std::sort(perm.begin(), perm.end(), [cp](int a, int b) {
    return cp->instance.device_names[a] < cp->instance.device_names[b];
  });
  std::vector<string> new_devs;
  std::vector<string> new_tasks;
  new_devs.reserve(cp->group.group_size);
  new_tasks.reserve(cp->group.group_size);
  for (int pi : perm) {
    new_devs.push_back(cp->instance.device_names[pi]);
    new_tasks.push_back(cp->instance.task_names[pi]);
  }
  cp->instance.device_names = std::move(new_devs);
  cp->instance.task_names = std::move(new_tasks);
  VLOG(1) << "Modified device_names on " << cp;
  SetDevPerTask(cp);
}
}  // namespace

void CollectiveParamResolverLocal::CompleteTaskIsLocal(const string& task_name,
                                                       CollectiveParams* cp) {
  cp->task.is_local.resize(cp->group.group_size, false);
  for (int i = 0; i < cp->group.group_size; ++i) {
    cp->task.is_local[i] = (cp->instance.task_names[i] == task_name);
  }
}

void CollectiveParamResolverLocal::SetDefaultRank(const string& device,
                                                  CollectiveParams* cp) {
  CHECK_EQ(cp->group.group_size, cp->instance.device_names.size()) << cp;
  for (int i = 0; i < cp->group.group_size; ++i) {
    if (cp->instance.device_names[i] == device) {
      cp->default_rank = i;
      break;
    }
  }
}

void CollectiveParamResolverLocal::InitInstanceSharedParams(
    const GroupRec* gr, const CollectiveParams* cp, InstanceRec* ir,
    const StatusCallback& done) {
  ir->shared.instance = cp->instance;
  {
    mutex_lock gl(gr->mu);
    ir->shared.group = gr->group;
    ir->shared.instance.device_names.assign(gr->device_list.begin(),
                                            gr->device_list.end());
    ir->shared.instance.task_names.assign(gr->task_list.begin(),
                                          gr->task_list.end());
    VLOG(2) << "Initialized names for instance: "
            << ir->shared.instance.ToString();
  }
  ir->shared.default_rank = -1;

  // Sort device_names lexicographically, keeping task_names in corresponding
  // order.  Also set number of devices per task.
  SortDevicesAndTasks(&ir->shared);

  // Get Locality data for all devices.

  // Set is_local and task_names in *shared prior to invoking
  // GetDeviceAttributesAsync.  In a distributed context this function can be
  // called by a derived class, some of the devices may be non-local and
  // GetDeviceAttributesAsync will use those fields to launch RPCs.
  CompleteTaskIsLocal(task_name_, &ir->shared);

  // Because the callback may execute in a different thread, we release
  // ir->out_mu here.  Before releasing, we mark it as unavailable for other
  // threads.
  ir->out_mu_available = false;
  const auto device_names = ir->shared.instance.device_names;
  const auto task_names = ir->shared.instance.task_names;
  ir->out_mu.unlock();
  std::vector<DeviceAttributes>* attributes = new std::vector<DeviceAttributes>;
  // Suppress linter warning about access to shared without mutex because in
  // principle the members are locked due to out_mu_available=false.
  dev_resolver_->GetAllDeviceAttributesAsync(
      ir->shared.instance.device_names,  // NOLINT
      ir->shared.instance.task_names,    // NOLINT
      attributes,
      [this, gr, cp, ir, attributes, done](const Status& s)
          EXCLUSIVE_LOCK_FUNCTION(ir->out_mu) {
            // Then we recover the lock in the callback thread that will hold it
            // through the rest of the call chain.  Signal the cv now, any
            // waiting threads will wake only when out_mu is released later.
            ir->out_mu.lock();
            DCHECK(!ir->out_mu_available);
            ir->out_mu_available = true;
            ir->out_cv.notify_all();
            if (s.ok()) {
              CompleteDefaultRanking(gr, cp, ir, *attributes);
              done(Status::OK());
            } else {
              done(s);
            }
            delete attributes;
          });
}

// NOTE(ayushd): The DeviceLocality objects in attributes will have LocalLinks
// to all devices that they are physically connected to and visible to the
// TensorFlow runtime.  This set of devices may be a superset of the devices
// participating in this instance of collectives.
void CollectiveParamResolverLocal::CompleteDefaultRanking(
    const GroupRec* gr, const CollectiveParams* cp, InstanceRec* ir,
    const std::vector<DeviceAttributes>& attributes) {
  // Establish an instance-specific default rank order for devices
  // based on localities.  This rank order should be a good ring
  // order, if possible.
  GlobalDeviceMap gdm = EstablishGlobalRank(&ir->shared, attributes);
  // Reflect the new global ranking on shared
  size_t num_devices = ir->shared.group.group_size;
  std::vector<string> new_device_names(num_devices, "");
  std::vector<string> new_task_names(num_devices, "");
  for (const auto& git : gdm) {
    const TaskDeviceMap& tdm = git.second;
    for (const auto& tit : tdm) {
      const DevRec& dr = tit.second;
      new_device_names[dr.global_rank] =
          ir->shared.instance.device_names[dr.original_rank];
      new_task_names[dr.global_rank] =
          ir->shared.instance.task_names[dr.original_rank];
    }
  }

  ir->shared.instance.device_names = new_device_names;
  ir->shared.instance.task_names = new_task_names;
  if (VLOG_IS_ON(2)) {
    string buf;
    for (const auto& d : new_device_names) strings::StrAppend(&buf, "\n", d);
    VLOG(2) << "Optimized device order for " << ir->shared.name << ": " << buf;
  }
}

void CollectiveParamResolverLocal::CallbackWithStatus(
    const InstanceRecCallback& done, InstanceRec* irec) {
  Status s;
  {
    mutex_lock l(irec->out_mu);
    irec->WaitForOutMu(l);
    s = irec->status;
  }
  done(s, irec);
}

void CollectiveParamResolverLocal::FindInstanceRec(
    const GroupRec* gr, CollectiveParams* cp, const InstanceRecCallback& done) {
  InstanceRec* irec = nullptr;
  bool exit_outside_locks = false;
  {
    mutex_lock l(instance_mu_);
    auto it = instance_table_.find(cp->instance.instance_key);
    if (it != instance_table_.end()) {
      irec = it->second.get();
      {
        mutex_lock l(irec->in_mu);
        if (irec->is_init) {
          exit_outside_locks = true;
        } else {
          irec->init_waiters.push_back([this, done](InstanceRec* irec) {
            CallbackWithStatus(done, irec);
          });
          return;
        }
      }
    } else {
      // Create new InstanceRec.
      irec = new InstanceRec;
      instance_table_[cp->instance.instance_key].reset(irec);
    }
  }
  if (exit_outside_locks) {
    CallbackWithStatus(done, irec);
    return;
  }

  CallInitInstanceSharedParams(gr, cp, irec, done);
}

void CollectiveParamResolverLocal::CallInitInstanceSharedParams(
    const GroupRec* gr, const CollectiveParams* cp, InstanceRec* ir,
    const InstanceRecCallback& done) NO_THREAD_SAFETY_ANALYSIS {
  // This function serves merely to make a function call that should
  // be thread/mutex safe but violates the simple model applied by
  // static analysis, so we turn off analysis only within this
  // function body.
  //
  // A lock on ir->out_mu must be held* throughout the _bodies_ of the
  // chain of function calls initiated here, each of which calls
  // another as its last action, but it will be dropped within the
  // callback defined below, which means that the lock can be dropped
  // before all the function stack frames pop. The static analysis will
  // not allow that.
  //
  // *the lock is dropped just before calling GetDeviceAttributesAsync, because
  // there is no guarantee that the thread that executes the callback is the
  // same as the one that locked ir->out_mu.  To prevent other threads from
  // grabbing ir->out_mu, we mark ir->out_mu_available as false.  Hence, in
  // principle, the lock is held throughout.
  ir->out_mu.lock();
  DCHECK(ir->out_mu_available);
  ir->known.resize(cp->group.group_size, false);
  InitInstanceSharedParams(
      gr, cp, ir,
      [this, ir, done](const Status& s) UNLOCK_FUNCTION(ir->out_mu) {
        DCHECK(ir->out_mu_available);
        ir->status.Update(s);
        ir->out_mu.unlock();
        // Prepare to invoke any waiters that accumulated during
        // initialization.
        std::vector<IRConsumer> init_waiters;
        {
          mutex_lock tl(instance_mu_);
          {
            mutex_lock l(ir->in_mu);
            ir->is_init = true;
            if (!ir->init_waiters.empty()) {
              std::swap(init_waiters, ir->init_waiters);
            }
          }
        }
        CallbackWithStatus(done, ir);
        for (auto& f : init_waiters) {
          f(ir);
        }
      });
}

void CollectiveParamResolverLocal::CompleteParamsAsync(
    const string& device, CollectiveParams* cp, CancellationManager* cancel_mgr,
    const StatusCallback& done) {
  VLOG(1) << "CompleteParams local " << device << " for " << cp << ": "
          << cp->ToString();
  CompleteGroupLocal(
      device, cp,
      [this, device, cp, done](const Status& s, const GroupRec* gr) {
        if (s.ok()) {
          CompleteInstanceLocal(device, gr, cp, cp->is_source, done);
        } else {
          done(s);
        }
      });
}

void CollectiveParamResolverLocal::CompleteInstanceAsync(
    const CompleteInstanceRequest* request, CompleteInstanceResponse* response,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  done(
      errors::Internal("CompleteInstance is not implemented by "
                       "CollectiveParamResolverLocal which is "
                       "intended only for non-distributed deployment."));
}

// TODO(b/111897089): we need a better way to pick the collective
// implementation.  The ideal way would depend upon the topology and link
// strength before picking a particular implementation.
void CollectiveParamResolverLocal::AssignCollectiveType(CollectiveParams* cp) {
  // We use the NCCL implementation if this is an environment which supports
  // NCCL, i.e. `LookupParamResolverInstance` for `NcclReduce` returns OK, and
  // also if indicated either in `ConfigProto` or `communication_hint`.
  //
  // After enough testing, we may simplify this logic to use NCCL whenever
  // available.
  CollectiveImplementationInterface* col_impl;
  bool use_nccl =
      (nccl_ || cp->instance.impl_details.communication_hint == "nccl") &&
      CollectiveRegistry::LookupParamResolverInstance("NcclReduce", &col_impl)
          .ok();
  cp->instance.impl_details.collective_name = GetCollectiveName(cp, use_nccl);
  VLOG(1) << "AssignCollectiveType "
          << cp->instance.impl_details.collective_name;
}

void CollectiveParamResolverLocal::CompleteInstanceLocal(
    const string& device, const GroupRec* gr, CollectiveParams* cp,
    bool is_source, const StatusCallback& done) {
  VLOG(1) << "CompleteInstanceLocal " << device
          << " instance_key: " << cp->instance.instance_key << " gr " << gr;

  // Populate the group portion of *cp from *gr.  Most of it should already
  // match.
  DCHECK_EQ(cp->group.group_key, gr->group.group_key);
  DCHECK_EQ(cp->group.group_size, gr->group.group_size);
  DCHECK_EQ(cp->group.device_type, gr->group.device_type);
  cp->group = gr->group;

  // Get the shared InstanceRec for this instance.
  FindInstanceRec(gr, cp,
                  [this, device, gr, cp, is_source, done](const Status& s,
                                                          InstanceRec* ir) {
                    if (s.ok()) {
                      CompleteInstanceFromInitializedIRec(device, gr, cp, ir,
                                                          is_source, done);
                    } else {
                      done(s);
                    }
                  });
}

void CollectiveParamResolverLocal::CompleteInstanceFromInitializedIRec(
    const string& device, const GroupRec* gr, CollectiveParams* cp,
    InstanceRec* ir, bool is_source, const StatusCallback& done) {
  auto expected_shape = cp->instance.shape;
  // Populate the fields common across instance.
  {
    mutex_lock l(ir->out_mu);
    ir->WaitForOutMu(l);
    // custom operator= does a deep copy.
    cp->instance = ir->shared.instance;
  }
  if (expected_shape != cp->instance.shape) {
    done(errors::InvalidArgument(
        "Shape mismatch in the collective instance ", cp->instance.instance_key,
        ". Op at device ", device, " expected shape ",
        expected_shape.DebugString(), " but another member in the group ",
        "expected shape ", cp->instance.shape.DebugString(), ". This is likely",
        " due to different input shapes at different members of the collective",
        " op."));
    return;
  }
  // Populate the fields common across task.
  AssignCollectiveType(cp);
  SetDefaultRank(device, cp);
  CompleteTaskIsLocal(task_name_, cp);

  CollectiveImplementationInterface* col_impl;
  Status status = CollectiveRegistry::LookupParamResolverInstance(
      cp->instance.impl_details.collective_name, &col_impl);
  if (!status.ok()) {
    done(status);
    return;
  }

  //  We may need to wait for the group, if this is a broadcast, for source
  //  discovery.
  if (cp->instance.type == BROADCAST_COLLECTIVE) {
    WaitForGroup(ir, cp, is_source,
                 [col_impl, ir, device, cp, done](InstanceRec* irec) {
                   Status s;
                   if (ir != irec) {
                     s = errors::Internal("Expected ir ", ir, " and irec ",
                                          irec, " to be equal");
                   } else {
                     mutex_lock l(irec->out_mu);
                     irec->WaitForOutMu(l);
                     s = irec->status;
                     cp->source_rank = irec->source_rank;
                   }
                   if (s.ok()) {
                     s = col_impl->InitializeCollectiveParams(cp);
                   }
                   done(s);
                 });
  } else {
    done(col_impl->InitializeCollectiveParams(cp));
  }
}

void CollectiveParamResolverLocal::WaitForGroup(InstanceRec* ir,
                                                CollectiveParams* cp,
                                                bool is_source,
                                                const IRConsumer& f) {
  std::vector<IRConsumer> ready_waiters;
  {
    mutex_lock l(ir->out_mu);
    ir->WaitForOutMu(l);
    CHECK_EQ(cp->group.group_size, ir->known.size());
    CHECK_GE(cp->default_rank, 0);
    if (!ir->known[cp->default_rank]) {
      ir->known[cp->default_rank] = true;
      ++ir->known_count;
      if (is_source) {
        // Initialize source rank.
        if (ir->source_rank >= 0) {
          ir->status = errors::Internal("Instance ", cp->instance.instance_key,
                                        " already has source ", ir->source_rank,
                                        ", received second claim from ",
                                        cp->default_rank);
        } else {
          ir->source_rank = cp->default_rank;
        }
      }
    }
    if (ir->known_count < ir->shared.group.group_size) {
      ir->known_waiters.push_back(f);
      return;
    }
    CHECK_EQ(ir->known_count, ir->shared.group.group_size);
    if (ir->source_rank < 0) {
      // NOTE(ayushd): changing the error message below would also require
      // updating CompleteParamsBroadcastForgotSend test in
      // CollectiveParamResolverLocalTest.
      ir->status =
          errors::Internal("Instance ", cp->instance.instance_key,
                           " found no source for broadcast.  This "
                           "could mean that there were group_size=",
                           ir->known_count, " BcastRecvs but no BcastSend.");
    }
    if (!ir->known_waiters.empty()) {
      ready_waiters = std::move(ir->known_waiters);
    }
  }
  f(ir);
  for (auto& f : ready_waiters) {
    f(ir);
  }
}

}  // namespace tensorflow
