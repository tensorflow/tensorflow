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

#include "tensorflow/core/common_runtime/device_mgr.h"

namespace tensorflow {

void CollectiveParamResolverLocal::InstanceRec::WaitForOutMu(mutex_lock& lock) {
  while (!out_mu_available) out_cv.wait(lock);
}

CollectiveParamResolverLocal::CollectiveParamResolverLocal(
    const DeviceMgr* dev_mgr, DeviceResolverInterface* dev_resolver,
    const string& task_name)
    : dev_mgr_(dev_mgr), dev_resolver_(dev_resolver), task_name_(task_name) {}

void CollectiveParamResolverLocal::CompleteGroupAsync(
    const CompleteGroupRequest* request, CompleteGroupResponse* response,
    CancellationManager* cancel_mgr, const StatusCallback& done) {
  done(
      errors::Internal("CompleteGroup is not implemented by "
                       "CollectiveParamResolverLocal which is "
                       "intended only for non-distributed deployment."));
}

void CollectiveParamResolverLocal::CompleteGroupLocal(
    const string& device, CollectiveParams* cp, const GroupRecCallback& done) {
  VLOG(1) << "CompleteGroupLocal device=" << device << " cp: " << cp << ": "
          << cp->ToString();
  std::vector<StatusCallback> to_be_called;
  GroupRec* gr = nullptr;
  {
    mutex_lock l(group_mu_);
    auto it = group_table_.find(cp->group.group_key);
    if (it == group_table_.end()) {
      gr = new GroupRec;
      gr->group.group_key = cp->group.group_key;
      gr->group.group_size = cp->group.group_size;
      gr->group.device_type = cp->group.device_type;
      group_table_[gr->group.group_key].reset(gr);
      VLOG(2) << "New group_key=" << gr->group.group_key
              << " group_size=" << gr->group.group_size;
    } else {
      gr = it->second.get();
    }
  }
  Status status;
  {
    mutex_lock gr_lock(gr->mu);
    if (!gr->device_set.empty()) {
      // Check for consistency with existing GroupRec.
      if (cp->group.device_type != gr->group.device_type) {
        status = errors::Internal(
            "Collective Op ", cp->name, " is assigned to device ", device,
            " with type ", cp->group.device_type.type_string(),
            " and group_key ", cp->group.group_key, " but that group has type ",
            gr->group.device_type.type_string());
      } else if (cp->group.group_size != gr->group.group_size) {
        status = errors::Internal(
            "Collective Op ", cp->name, " has group_size ",
            cp->group.group_size, " and group_key", cp->group.group_key,
            " but that group has size ", gr->group.group_size);
      }
    }
    if (status.ok()) {
      // Insert device if not already present.
      auto it = gr->device_set.find(device);
      if (it == gr->device_set.end()) {
        if (gr->device_set.size() == gr->group.group_size) {
          // The group is already full.
          status = errors::Internal(
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
          VLOG(1) << "group_key=" << gr->group.group_key
                  << " group_size=" << gr->group.group_size
                  << " dev_set=" << gr->device_set.size();
        }
      }
    }

    if (status.ok()) {
      // If the group is not yet complete, queue to wait for it.
      VLOG(2) << "group_size " << gr->group.group_size << " set size "
              << gr->device_set.size() << " gr " << gr;

      if (gr->device_set.size() < gr->group.group_size) {
        gr->waiting.push_back(std::bind(done, std::placeholders::_1, gr));
        return;
      }
      CHECK_EQ(gr->device_set.size(), gr->group.group_size);
      if (!gr->waiting.empty()) {
        std::swap(to_be_called, gr->waiting);
      }
    }
  }
  done(status, gr);
  for (int i = 0; i < to_be_called.size(); ++i) {
    to_be_called[i](Status::OK());
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
                             const std::vector<DeviceLocality>& localities) {
  GlobalDeviceMap gdm;
  CHECK_EQ(ip.device_names.size(), ip.task_names.size());
  CHECK_EQ(ip.device_names.size(), localities.size());
  for (int i = 0; i < ip.device_names.size(); ++i) {
    TaskDeviceMap& tdm = gdm[ip.task_names[i]];
    DevRec* dr = &tdm[ip.device_names[i]];
    dr->task = ip.task_names[i];
    dr->device = ip.device_names[i];
    dr->original_rank = i;
    dr->local_rank = 0;   // Will be populated later by OrderTaskDeviceMap.
    dr->global_rank = 0;  // Will be populated later by EstablishGlobalRank.
    dr->locality = &localities[i];
  }
  return gdm;
}

void OrderTaskDeviceMap(TaskDeviceMap* tdm) {
  CHECK_GT(tdm->size(), 0);  // Should never be called with 0 devices
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
    DevRec* dr = &(*tdm)[next_device];
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
        if (selected.find(endpoint_device) != selected.end()) {
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
    CollectiveParams* cp, const std::vector<DeviceLocality>& localities) {
  VLOG(1) << "EstablishGlobalRank";
  GlobalDeviceMap gdm = BuildDevRecs(cp->instance, localities);
  for (auto& iter : gdm) {
    TaskDeviceMap& tdm = iter.second;
    OrderTaskDeviceMap(&tdm);
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
  cp->instance.same_num_devices_per_task = false;
  if (cp->instance.task_names.empty()) return;
  int dev_per_task = -1;
  int count = 0;
  const string* last_task_name = &cp->instance.task_names[0];
  for (const string& task_name : cp->instance.task_names) {
    if (task_name != *last_task_name) {
      CHECK_GT(count, 0);
      if (dev_per_task < 0) {
        dev_per_task = count;
      } else {
        CHECK_GT(dev_per_task, 0);
        if (count != dev_per_task) return;
      }
      count = 1;
      last_task_name = &task_name;
    } else {
      ++count;
    }
  }
  CHECK_GT(count, 0);
  if ((dev_per_task > 0) && (count != dev_per_task)) {
    return;
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
  std::sort(perm.begin(), perm.end(), [cp](const int& a, const int& b) {
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

// Establish the requested number of subdivision permutations based on the
// ring order implicit in the device order.
void GenerateSubdivPerms(const string& device, int source_rank,
                         CollectiveParams* cp) {
  // Each subdiv permutation is a ring formed by rotating each
  // single-task subsequence of devices by an offset.  This makes most
  // sense when each task has the same number of devices but we can't
  // depend on that being the case so we'll compute something that
  // works in any case.

  // Start by counting the devices in each task.
  // Precondition: device_names must be sorted so that all devices in
  // the same task are adjacent.
  VLOG(2) << "Sorted task names: "
          << str_util::Join(cp->instance.task_names, ", ");
  std::vector<int> dev_per_task;
  const string* prior_task_name = &cp->instance.task_names[0];
  int dev_count = 1;
  for (int di = 1; di < cp->group.group_size; ++di) {
    if (cp->instance.task_names[di] != *prior_task_name) {
      dev_per_task.push_back(dev_count);
      dev_count = 1;
      prior_task_name = &cp->instance.task_names[di];
    } else {
      ++dev_count;
    }
  }
  dev_per_task.push_back(dev_count);
  CHECK_EQ(cp->group.num_tasks, dev_per_task.size());

  // Generate a ring permutation for each requested offset.
  CHECK_GT(cp->instance.impl_details.subdiv_offsets.size(), 0);
  VLOG(2) << "Setting up perms for cp " << cp << " subdiv_permutations "
          << &cp->instance.impl_details.subdiv_permutations;
  cp->instance.impl_details.subdiv_permutations.resize(
      cp->instance.impl_details.subdiv_offsets.size());
  cp->subdiv_rank.resize(cp->instance.impl_details.subdiv_offsets.size(), -1);
  for (int sdi = 0; sdi < cp->instance.impl_details.subdiv_offsets.size();
       ++sdi) {
    std::vector<int>& perm = cp->instance.impl_details.subdiv_permutations[sdi];
    CHECK_EQ(perm.size(), 0);
    int offset = cp->instance.impl_details.subdiv_offsets[sdi];
    int prior_dev_count = 0;
    for (int ti = 0; ti < cp->group.num_tasks; ++ti) {
      for (int di = 0; di < dev_per_task[ti]; ++di) {
        int offset_di = (di + offset) % dev_per_task[ti];
        int permuted_di = prior_dev_count + offset_di;
        perm.push_back(permuted_di);
        if (cp->instance.device_names[prior_dev_count + di] == device) {
          CHECK_EQ(prior_dev_count + di, cp->default_rank);
          cp->subdiv_rank[sdi] = permuted_di;
        }
      }
      prior_dev_count += dev_per_task[ti];
    }
    CHECK_EQ(cp->group.group_size, perm.size());
  }

  if (cp->instance.type == BROADCAST_COLLECTIVE) {
    CHECK_GE(source_rank, 0);
    cp->instance.impl_details.subdiv_source_rank.resize(
        cp->instance.impl_details.subdiv_offsets.size(), -1);
    for (int sdi = 0; sdi < cp->instance.impl_details.subdiv_source_rank.size();
         ++sdi) {
      for (int j = 0; j < cp->group.group_size; ++j) {
        if (cp->instance.impl_details.subdiv_permutations[sdi][j] ==
            source_rank) {
          cp->instance.impl_details.subdiv_source_rank[sdi] = j;
          break;
        }
      }
      CHECK_GE(cp->instance.impl_details.subdiv_source_rank[sdi], 0);
    }
  }

  if (VLOG_IS_ON(1)) {
    // Log the computed ring order for each subdiv.
    string buf;
    for (int sdi = 0;
         sdi < cp->instance.impl_details.subdiv_permutations.size(); ++sdi) {
      buf = strings::StrCat("Subdiv ", sdi, " device order:\n");
      for (int di = 0;
           di < cp->instance.impl_details.subdiv_permutations[sdi].size();
           ++di) {
        int idx = cp->instance.impl_details.subdiv_permutations[sdi][di];
        strings::StrAppend(&buf, cp->instance.device_names[idx], "\n");
      }
      strings::StrAppend(&buf, " subdiv_offsets: ");
      for (auto o : cp->instance.impl_details.subdiv_offsets)
        strings::StrAppend(&buf, o, " ");
      strings::StrAppend(&buf, " SubdivRank: ");
      for (auto d : cp->subdiv_rank) strings::StrAppend(&buf, d, " ");
      VLOG(1) << buf;
    }
  }
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
  VLOG(1) << "InitInstanceSharedParams " << ir;
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

  // Sort devce_names lexicographcally, keeping task_names in
  // corresponding order.
  SortDevicesAndTasks(&ir->shared);

  // Get Locality data for all devices.

  // Set is_local and task_names in *shared prior to invoking
  // GetDeviceLocalitiesAsync.  In a distributed context this function can be
  // called by a derived class, some of the devices may be non-local and
  // GetDeviceLocalitiesAsync will use those fields to launch RPCs.
  CompleteTaskIsLocal(task_name_, &ir->shared);

  // Because the callback may execute in a different thread, we release
  // ir->out_mu here.  Before releasing, we mark it as unavailable for other
  // threads.
  ir->out_mu_available = false;
  ir->out_mu.unlock();
  std::vector<DeviceLocality>* localities = new std::vector<DeviceLocality>;
  dev_resolver_->GetDeviceLocalitiesAsync(
      ir->shared.instance, localities,
      [this, gr, cp, ir, localities, done](const Status& s)
          EXCLUSIVE_LOCK_FUNCTION(ir->out_mu) {
            // Then we recover the lock in the callback thread that will hold it
            // through the rest of the call chain.  Signal the cv now, any
            // waiting threads will wake only when out_mu is released later.
            ir->out_mu.lock();
            DCHECK(!ir->out_mu_available);
            ir->out_mu_available = true;
            ir->out_cv.notify_all();
            if (s.ok()) {
              CompleteDefaultRanking(gr, cp, ir, *localities);
              done(Status::OK());
            } else {
              done(s);
            }
            delete localities;
          });
}

void CollectiveParamResolverLocal::CompleteDefaultRanking(
    const GroupRec* gr, const CollectiveParams* cp, InstanceRec* ir,
    const std::vector<DeviceLocality>& localities) {
  // Establish an instance-specific default rank order for devices
  // based on localities.  This rank order should be a good ring
  // order, if possible.
  GlobalDeviceMap gdm = EstablishGlobalRank(&ir->shared, localities);
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
    for (const auto& d : cp->instance.device_names)
      strings::StrAppend(&buf, "\n", d);
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
          irec->init_waiters.push_back([this, gr, cp, done](InstanceRec* irec) {
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
  // *the lock is dropped just before calling GetDeviceLocalitiesAsync, because
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
        DCHECK(!ir->out_mu.try_lock());
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
  VLOG(1) << "CompleteParams " << device << " for " << cp << ": "
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
  // Populate the fields common across instance.
  {
    mutex_lock l(ir->out_mu);
    ir->WaitForOutMu(l);
    // custom operator= does a deep copy.
    cp->instance = ir->shared.instance;
  }
  // Populate the fields common across task, also default_rank.
  SetDefaultRank(device, cp);
  CompleteTaskIsLocal(task_name_, cp);
  // If broadcast, may need to wait for source discovery.
  if (cp->instance.type == BROADCAST_COLLECTIVE) {
    CompleteInstanceSource(ir, cp, is_source,
                           [this, ir, device, cp, done](InstanceRec* irec) {
                             CHECK_EQ(ir, irec);
                             Status s;
                             int source_rank;
                             {
                               mutex_lock l(irec->out_mu);
                               irec->WaitForOutMu(l);
                               s = irec->status;
                               source_rank = irec->source_rank;
                             }
                             if (s.ok()) {
                               GenerateSubdivPerms(device, source_rank, cp);
                             }
                             done(s);
                           });
    return;
  } else {
    GenerateSubdivPerms(device, 0, cp);
  }
  done(Status::OK());
}

void CollectiveParamResolverLocal::CompleteInstanceSource(InstanceRec* ir,
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
        if (ir->source_rank >= 0) {
          ir->status = errors::Internal("Instance ", cp->instance.instance_key,
                                        " already has source ", ir->source_rank,
                                        ", recevied second claim from ",
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
    CHECK_GE(ir->source_rank, 0);
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
