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
#include <unordered_set>
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
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

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
      return nccl ? "NcclBroadcast" : "HierarchicalTreeBroadcast";

    case REDUCTION_COLLECTIVE:
      return nccl ? "NcclReduce" : "RingReduce";

    case GATHER_COLLECTIVE:
      return nccl ? "NcclGather" : "RingGather";

    case PERMUTE_COLLECTIVE:
      return "Permute";

    default:
      return "undef";
  }
}

string TaskNameFromDeviceName(const string& device_name) {
  DeviceNameUtils::ParsedName parsed_device;
  CHECK(DeviceNameUtils::ParseFullName(device_name, &parsed_device));
  string task_name;
  CHECK(DeviceNameUtils::GetTaskName(parsed_device, &task_name));
  return task_name;
}
}  // namespace

void CollectiveParamResolverLocal::CompleteGroupLocal(
    const DeviceAttributes& device, CollectiveParams* cp,
    const GroupRecCallback& done, CancellationManager* cancel_mgr) {
  VLOG(1) << "CompleteGroupLocal device=" << device.name() << " cp: " << cp
          << ": " << cp->ToString();
  std::vector<StatusCallback> to_be_called;
  // Keep a reference to `cp` to avoid racing with deletion due to cancellation.
  cp->Ref();
  core::ScopedUnref cp_unref(cp);

  std::function<void(const Status& s, GroupRec* gr)> done_with_cleanup;
  if (cancel_mgr != nullptr) {
    auto cancelled_mu = std::make_shared<mutex>();
    // Some callers delete `cancel_mgr` as soon as `done` is called once,
    // meaning we can't rely on it to avoid calling `done` twice if the local op
    // is cancelled but the group succeeds.
    auto cancelled = std::make_shared<bool>(false);
    const CancellationToken token = cancel_mgr->get_cancellation_token();
    const bool already_cancelled =
        !cancel_mgr->RegisterCallback(token, [done, cancelled_mu, cancelled]() {
          {
            mutex_lock l(*cancelled_mu);
            *cancelled = true;
          }
          done(errors::Cancelled("op cancelled"), nullptr);
        });
    if (already_cancelled) {
      done(errors::Cancelled("op cancelled"), nullptr);
      return;
    }
    done_with_cleanup = [cancel_mgr, done, cancelled_mu, cancelled, token](
                            const Status& s, GroupRec* gr) {
      {
        mutex_lock l(*cancelled_mu);
        if (*cancelled || !cancel_mgr->TryDeregisterCallback(token)) {
          return;
        }
      }
      // The operation was never cancelled, so we'll return a normal status.
      done(s, gr);
    };
  } else {
    done_with_cleanup = done;
  }

  GroupRec* gr = nullptr;
  Status status;
  {
    mutex_lock l(group_mu_);
    auto it = group_table_.find(cp->group.group_key);
    if (it == group_table_.end()) {
      gr = new GroupRec;
      mutex_lock grl(gr->mu);
      gr->group.group_key = cp->group.group_key;
      gr->group.group_size = cp->group.group_size;
      gr->group.device_type = cp->group.device_type;
      gr->group.gpu_ring_order = cp->group.gpu_ring_order;

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
        done_with_cleanup(status, gr);
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
    mutex_lock l(status_mu_);
    status = status_;
  }
  if (!status.ok()) {
    done_with_cleanup(status, nullptr);
    return;
  }
  {
    mutex_lock gr_lock(gr->mu);
    // If there is ever an error associated with a group key, we store the error
    // status and invoke all waiting and future callbacks with this error
    // status.
    VLOG(2) << "gr device_type=" << gr->group.device_type
            << " cp device_type=" << cp->group.device_type
            << " current device=" << device.name();
    if (gr->status.ok()) {
      // Check for consistency with existing GroupRec.
      if (cp->group.device_type != gr->group.device_type) {
        gr->status = errors::Internal(
            "Collective Op ", cp->name, " is assigned to device ",
            device.name(), " with type ", cp->group.device_type.type_string(),
            " and group_key ", cp->group.group_key, " but that group has type ",
            gr->group.device_type.type_string());
      } else if (cp->group.group_size != gr->group.group_size) {
        gr->status = errors::Internal(
            "Collective Op ", cp->name, " has group_size ",
            cp->group.group_size, " and group_key ", cp->group.group_key,
            " but that group has size ", gr->group.group_size);
      }
    }
    bool new_device = false;
    if (gr->status.ok()) {
      // Insert device if not already present.
      auto it = gr->devices.find(device.name());
      if (it == gr->devices.end()) {
        if (gr->devices.size() == gr->group.group_size) {
          // The group is already full.
          gr->status = errors::Internal(
              "Collective Op ", cp->name, " is assigned to device ",
              device.name(), " and group_key ", cp->group.group_key,
              " but that group doesn't contain that device.");
        } else {
          // This is a new device that has not yet joined the group.
          gr->devices[device.name()] = device;
          new_device = true;
          if (VLOG_IS_ON(1)) {
            string dev_buf;
            for (const auto& d : gr->devices) {
              strings::StrAppend(&dev_buf, ",", d.first);
            }
            VLOG(1) << "CompleteGroupLocal group_key=" << gr->group.group_key
                    << " group_size=" << gr->group.group_size << " (current"
                    << " devices)=(" << dev_buf << ") (number of"
                    << " devices pending)="
                    << (gr->group.group_size - gr->devices.size());
          }
        }
      } else {
        // If the device already exists, check if the incarnation matches.
        if (it->second.incarnation() != device.incarnation()) {
          gr->status = errors::FailedPrecondition(
              "Device ", device.name(),
              " current incarnation doesn't match with one in the group. This "
              "usually means this worker has restarted but the collective "
              "leader hasn't, or this worker connects to a wrong cluster.");
        }
      }
    }

    if (gr->status.ok()) {
      // If the group is not yet complete, queue to wait for it.
      VLOG(2) << "group_size " << gr->group.group_size << " set size "
              << gr->devices.size() << " gr " << gr;

      if (gr->devices.size() < gr->group.group_size) {
        gr->waiting.push_back(
            std::bind(done_with_cleanup, std::placeholders::_1, gr));
        return;
      }
      CHECK_EQ(gr->devices.size(), gr->group.group_size);
      // We get a full group. Fill in remaining fields in gr->group.
      if (new_device) {
        FinishGroup(gr);
      }
    }
    // At this point, we either have a full group, or an error status.  Ensure
    // that all callbacks are invoked with the appropriate status.
    if (!gr->waiting.empty()) {
      std::swap(to_be_called, gr->waiting);
    }
    status = gr->status;
  }
  done_with_cleanup(status, gr);
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
GlobalDeviceMap BuildDevRecs(const CollGroupParams& gp,
                             const std::vector<DeviceAttributes>& attributes) {
  GlobalDeviceMap gdm;
  CHECK_EQ(gp.device_names.size(), gp.task_names.size());
  CHECK_EQ(gp.device_names.size(), attributes.size());
  for (int i = 0; i < gp.device_names.size(); ++i) {
    TaskDeviceMap& tdm = gdm[gp.task_names[i]];
    DevRec* dr = &tdm[gp.device_names[i]];
    dr->task = gp.task_names[i];
    dr->device = gp.device_names[i];
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

// The first time a CollGroupParams is established for a group we compute a good
// rank order for all the devices in the group, that is appropriate for a ring
// algorithm.
GlobalDeviceMap EstablishGlobalRank(
    const CollGroupParams& gp,
    const std::vector<DeviceAttributes>& attributes) {
  VLOG(1) << "EstablishGlobalRank";
  GlobalDeviceMap gdm = BuildDevRecs(gp, attributes);
  for (auto& iter : gdm) {
    TaskDeviceMap& tdm = iter.second;
    OrderTaskDeviceMap(gp.gpu_ring_order, &tdm);
  }
  // Connect the global rank order by the order in which tasks first appear.
  std::set<string> ordered_tasks;
  int next_rank = 0;
  for (int i = 0; i < gp.task_names.size(); ++i) {
    const string& task_name = gp.task_names[i];
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
// gp->same_num_devices_per_task.  Requires gp->task_names
// be sorted.
void SetDevPerTask(CollGroupParams* gp) {
  gp->num_devices_per_task.clear();
  const string* last_task_name = &gp->task_names[0];
  int count = 0;
  for (const string& task_name : gp->task_names) {
    if (task_name == *last_task_name) {
      ++count;
    } else {
      gp->num_devices_per_task[*last_task_name] = count;
      count = 1;
      last_task_name = &task_name;
    }
  }
  gp->num_devices_per_task[*last_task_name] = count;

  gp->same_num_devices_per_task = false;
  int dev_per_task = -1;
  for (const auto& task_dev : gp->num_devices_per_task) {
    if (dev_per_task == -1) {
      dev_per_task = task_dev.second;
    } else if (dev_per_task != task_dev.second) {
      return;
    }
  }
  gp->same_num_devices_per_task = true;
  CHECK_EQ((gp->group_size % gp->num_tasks), 0);
}

// Sort gp->device_names lexicographically, but do by first
// computing a reordering permutation so we can keep gp->task_names
// in corresponding order.
void SortDevicesAndTasks(CollGroupParams* gp) {
  VLOG(1) << "SortDevicesAndTasks " << gp << " " << gp;
  CHECK(gp);
  CHECK_EQ(gp->group_size, gp->device_names.size());
  CHECK_EQ(gp->group_size, gp->task_names.size());
  std::vector<int> perm(gp->group_size);
  // TODO(tucker): substitute std::iota when the windows build supports it.
  // std::iota(perm.begin(), perm.end(), 0);
  for (int i = 0; i < perm.size(); ++i) {
    perm[i] = i;
  }
  std::sort(perm.begin(), perm.end(), [gp](int a, int b) {
    return gp->device_names[a] < gp->device_names[b];
  });
  std::vector<string> new_devs;
  std::vector<string> new_tasks;
  new_devs.reserve(gp->group_size);
  new_tasks.reserve(gp->group_size);
  for (int pi : perm) {
    new_devs.push_back(gp->device_names[pi]);
    new_tasks.push_back(gp->task_names[pi]);
  }
  gp->device_names = std::move(new_devs);
  gp->task_names = std::move(new_tasks);
  VLOG(1) << "Modified device_names on " << gp;
  SetDevPerTask(gp);
}
}  // namespace

void CollectiveParamResolverLocal::FinishGroup(GroupRec* gr) {
  gr->group.device_names.reserve(gr->devices.size());
  gr->group.task_names.reserve(gr->devices.size());
  std::vector<DeviceAttributes> attributes;
  // Unique tasks. It's used to calculate num_tasks.
  std::unordered_set<string> tasks;
  attributes.reserve(gr->devices.size());
  for (const auto& item : gr->devices) {
    gr->group.device_names.push_back(item.first);
    string task_name = TaskNameFromDeviceName(item.first);
    gr->group.task_names.push_back(task_name);
    tasks.insert(task_name);
    attributes.push_back(item.second);
  }
  gr->group.num_tasks = static_cast<int32>(tasks.size());
  // Sort device_names lexicographically, keeping task_names in corresponding
  // order. Also set number of devices per task.
  SortDevicesAndTasks(&gr->group);
  // Establish the final order of gp->device_names and gp->task_names by
  // considering localities of all devices.
  CompleteDefaultRanking(attributes, &gr->group);
}

void CollectiveParamResolverLocal::CompleteTaskIsLocal(const string& task_name,
                                                       CollectiveParams* cp) {
  cp->task.is_local.resize(cp->group.group_size, false);
  for (int i = 0; i < cp->group.group_size; ++i) {
    cp->task.is_local[i] = (cp->group.task_names[i] == task_name);
  }
}

void CollectiveParamResolverLocal::SetDefaultRank(const string& device,
                                                  CollectiveParams* cp) {
  CHECK_EQ(cp->group.group_size, cp->group.device_names.size()) << cp;
  for (int i = 0; i < cp->group.group_size; ++i) {
    if (cp->group.device_names[i] == device) {
      cp->default_rank = i;
      break;
    }
  }
}

void CollectiveParamResolverLocal::InitInstanceSharedParams(
    const GroupRec* gr, const CollectiveParams* cp, InstanceRec* ir) {
  ir->shared->instance = cp->instance;
  ir->shared->default_rank = -1;

  // Set is_local and task_names in *shared prior to invoking
  // GetDeviceAttributesAsync.  In a distributed context this function can be
  // called by a derived class, some of the devices may be non-local and
  // GetDeviceAttributesAsync will use those fields to launch RPCs.
  CompleteTaskIsLocal(task_name_, ir->shared);
}

// NOTE(ayushd): The DeviceLocality objects in attributes will have LocalLinks
// to all devices that they are physically connected to and visible to the
// TensorFlow runtime.  This set of devices may be a superset of the devices
// participating in this instance of collectives.
void CollectiveParamResolverLocal::CompleteDefaultRanking(
    const std::vector<DeviceAttributes>& attributes, CollGroupParams* gp) {
  // Establish an instance-specific default rank order for devices
  // based on localities.  This rank order should be a good ring
  // order, if possible.
  GlobalDeviceMap gdm = EstablishGlobalRank(*gp, attributes);
  // Reflect the new global ranking on shared
  size_t num_devices = gp->group_size;
  std::vector<string> new_device_names(num_devices, "");
  std::vector<string> new_task_names(num_devices, "");
  for (const auto& git : gdm) {
    const TaskDeviceMap& tdm = git.second;
    for (const auto& tit : tdm) {
      const DevRec& dr = tit.second;
      new_device_names[dr.global_rank] = gp->device_names[dr.original_rank];
      new_task_names[dr.global_rank] = gp->task_names[dr.original_rank];
    }
  }

  gp->device_names = new_device_names;
  gp->task_names = new_task_names;
  if (VLOG_IS_ON(2)) {
    string buf;
    for (const auto& d : new_device_names) strings::StrAppend(&buf, "\n", d);
    VLOG(2) << "Optimized device order for group " << gp->group_key << ": "
            << buf;
  }
}

CollectiveParamResolverLocal::InstanceRec*
CollectiveParamResolverLocal::GetOrCreateInstanceRec(const GroupRec* gr,
                                                     CollectiveParams* cp,
                                                     bool* created) {
  *created = false;
  InstanceRec* irec = nullptr;
  {
    mutex_lock l(instance_mu_);
    auto group_it = instance_table_.find(gr->group.group_key);
    if (group_it != instance_table_.end()) {
      auto instance_it = group_it->second.find(cp->instance.instance_key);
      if (instance_it != group_it->second.end()) {
        irec = instance_it->second.get();
      }
    }
    if (irec == nullptr) {
      // Create new InstanceRec.
      irec = new InstanceRec;
      *created = true;
      {
        mutex_lock il(irec->mu);
        irec->known.resize(cp->group.group_size, false);
      }
      InitInstanceSharedParams(gr, cp, irec);
      instance_table_[gr->group.group_key][cp->instance.instance_key].reset(
          irec);
    }
  }
  Status status;
  {
    mutex_lock l(status_mu_);
    status = status_;
  }
  if (!status.ok()) {
    mutex_lock l(irec->mu);
    irec->status = status;
  }
  return irec;
}

void CollectiveParamResolverLocal::CompleteParamsAsync(
    const DeviceAttributes& device, CollectiveParams* cp,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  VLOG(1) << "CompleteParams local " << device.name() << " for " << cp << ": "
          << cp->ToString();
  CompleteGroupLocal(
      device, cp,
      [this, device, cp, done](const Status& s, const GroupRec* gr) {
        if (s.ok()) {
          CompleteInstanceLocal(device.name(), gr, cp, cp->is_source, done);
        } else {
          done(s);
        }
      },
      cancel_mgr);
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
  {
    mutex_lock l(gr->mu);
    DCHECK_EQ(cp->group.group_key, gr->group.group_key);
    DCHECK_EQ(cp->group.group_size, gr->group.group_size);
    DCHECK_EQ(cp->group.device_type, gr->group.device_type);
    cp->group = gr->group;
  }

  bool created_irec;
  InstanceRec* ir = GetOrCreateInstanceRec(gr, cp, &created_irec);
  if (!created_irec) {
    // Check that the preexisting IRec is consistent with the params passed into
    // this invocation.
    if (ir->shared->instance.type != cp->instance.type ||
        ir->shared->instance.data_type != cp->instance.data_type) {
      done(errors::Internal("Collective instance ", cp->instance.instance_key,
                            " expected type ", ir->shared->instance.type,
                            " and data_type ", ir->shared->instance.data_type,
                            " but got type ", cp->instance.type,
                            " and data_type ", cp->instance.data_type));
      return;
    }
  }
  CompleteInstanceFromInitializedIRec(device, gr, cp, ir, is_source, done);
}

void CollectiveParamResolverLocal::CompleteInstanceFromInitializedIRec(
    const string& device, const GroupRec* gr, CollectiveParams* cp,
    InstanceRec* ir, bool is_source, const StatusCallback& done) {
  auto expected_shape = cp->instance.shape;
  Status status;
  // Populate the fields common across instance.
  {
    mutex_lock l(ir->mu);
    status = ir->status;
    if (status.ok()) {
      // custom operator= does a deep copy.
      cp->instance = ir->shared->instance;
    }
  }
  if (!status.ok()) {
    done(status);
    return;
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
  status = CollectiveRegistry::LookupParamResolverInstance(
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
                     mutex_lock l(irec->mu);
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
  do {
    mutex_lock l(ir->mu);
    if (!ir->status.ok()) {
      break;
    }
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
    if (ir->known_count < cp->group.group_size) {
      ir->known_waiters.push_back(f);
      return;
    }
    CHECK_EQ(ir->known_count, cp->group.group_size);
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
  } while (false);
  f(ir);
  for (auto& f : ready_waiters) {
    f(ir);
  }
}

void CollectiveParamResolverLocal::StartAbort(const Status& s) {
  {
    mutex_lock l(status_mu_);
    if (!status_.ok()) {
      VLOG(2) << "CollectiveParamResolverLocal already aborted. Ignoring "
                 "subsequent abortion with status: "
              << s;
      return;
    }
    status_ = s;
  }
  StartAbortLocal(s);
}

void CollectiveParamResolverLocal::StartAbortLocal(const Status& s) {
  {
    mutex_lock l(group_mu_);
    for (const auto& item : group_table_) {
      GroupRec* gr = item.second.get();
      std::vector<StatusCallback> waiting;
      {
        mutex_lock gl(gr->mu);
        gr->status = s;
        waiting.swap(gr->waiting);
      }
      for (const StatusCallback& done : waiting) {
        done(s);
      }
    }
  }
  std::vector<InstanceRec*> instances;
  {
    mutex_lock l(instance_mu_);
    for (const auto& group_entry : instance_table_) {
      for (const auto& item : group_entry.second) {
        instances.push_back(item.second.get());
      }
    }
  }
  for (InstanceRec* ir : instances) {
    std::vector<IRConsumer> known_waiters;
    {
      mutex_lock il(ir->mu);
      ir->status = s;
      known_waiters.swap(ir->known_waiters);
    }
    for (const IRConsumer& done : known_waiters) {
      done(ir);
    }
  }
}

}  // namespace tensorflow
