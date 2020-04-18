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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_PARAM_RESOLVER_LOCAL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_PARAM_RESOLVER_LOCAL_H_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
class CompleteGroupRequest;
class CompleteGroupResponse;
class CompleteInstanceRequest;
class CompleteInstanceResponse;
class DeviceMgr;

// Implements ParamResolverInterface for a single-task context.
// It also implements the functionality necessary to serve as the
// group leader for param resolution in a multi-task context.
class CollectiveParamResolverLocal : public ParamResolverInterface {
 public:
  CollectiveParamResolverLocal(const ConfigProto& config,
                               const DeviceMgr* dev_mgr,
                               DeviceResolverInterface* dev_resolver,
                               const string& task_name);

  ~CollectiveParamResolverLocal() override {}

  void CompleteParamsAsync(const string& device, CollectiveParams* cp,
                           CancellationManager* cancel_mgr,
                           const StatusCallback& done) override;

  void CompleteGroupAsync(const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          CancellationManager* cancel_mgr,
                          const StatusCallback& done) override;

  void CompleteInstanceAsync(const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             CancellationManager* cancel_mgr,
                             const StatusCallback& done) override;

 protected:
  // For access to InstanceRec and CompleteDefaultRanking.
  friend class CollectiveParamResolverLocalTest;

  // Used to complete/verify CollGroup.
  struct GroupRec {
    CollGroupParams group;
    mutable mutex mu;
    Status status TF_GUARDED_BY(mu);
    std::set<string> device_set TF_GUARDED_BY(mu);
    std::vector<string> device_list TF_GUARDED_BY(mu);
    std::set<string> task_set TF_GUARDED_BY(mu);
    std::vector<string> task_list TF_GUARDED_BY(mu);
    std::vector<StatusCallback> waiting TF_GUARDED_BY(mu);
  };

  // Finds the GroupRec that corresponds to cp->group_key.
  // Also populates cp->group from that group_rec.
  // Will wait until GroupRec is fully populated or an error arises before
  // calling done.  Callback GroupRec* arg is only valid if status is ok.
  // Ownership of GroupRec stays with this object and does not pass to the
  // callback.
  typedef std::function<void(const Status& s, const GroupRec* gr)>
      GroupRecCallback;
  void CompleteGroupLocal(const string& device, CollectiveParams* cp,
                          const GroupRecCallback& done)
      TF_LOCKS_EXCLUDED(group_mu_);

  // Used to complete/verify CollInstance.
  struct InstanceRec;

  typedef std::function<void(InstanceRec*)> IRConsumer;
  struct InstanceRec {
    // This structure has two mutexes so that a possibly long
    // initialization can be done without holding the instance_mu_
    // table lock the whole time (which can cause an excessive number
    // of threads to block on it), and because the compiler may not
    // permit mutex locks to be taken in more than one order.
    //
    // out_mu guards access to most of the fields.
    // in_mu guards access to a queue of consumer callbacks wanting to
    // read the fields guarded by out_mu.
    //
    // The in_mu should be locked only while holding instance_mu_; the
    // out_mu should be locked only while not holding
    // instance_mu_.
    //
    // When is_init is false (the initial value) any potential user
    // other than the creator should queue a callback on init_waiters.
    // As soon as the shared member of this structure is fully
    // initialized is_init will be set true and those callbacks will
    // be invoked.
    //
    // Once inserted in the table this structure will never be replaced
    // so users can capture the pointer while holding instance_mu_,
    // drop that lock, then take a lock on out_mu before
    // reading/modifying its values.
    mutex in_mu;
    bool is_init TF_GUARDED_BY(in_mu);
    std::vector<IRConsumer> init_waiters TF_GUARDED_BY(in_mu);

    // A thread that wishes to acquire out_mu must ensure that it is available
    // by invoking WaitForOutMu().
    mutex out_mu;
    condition_variable out_cv;
    bool out_mu_available TF_GUARDED_BY(out_mu);
    // Values to be shared by all instances, constant after initialization.
    CollectiveParams shared TF_GUARDED_BY(out_mu);
    // If an error occurs during initialization this structure stays in
    // the table with a non-OK status.  Purging the table and restarting
    // needs to be done at a higher level.
    Status status TF_GUARDED_BY(out_mu);

    // These fields are used to count the instances that have called
    // in and become known while resolving broadcast source identity and
    // communicator key.
    int source_rank TF_GUARDED_BY(out_mu);
    string communicator_key TF_GUARDED_BY(out_mu);
    int known_count TF_GUARDED_BY(out_mu);
    std::vector<bool> known TF_GUARDED_BY(out_mu);
    std::vector<IRConsumer> known_waiters TF_GUARDED_BY(out_mu);

    InstanceRec()
        : is_init(false),
          out_mu_available(true),
          source_rank(-1),
          known_count(0) {}

    // If out_mu is unavailable during distributed device locality
    // initialization, wait on out_cv until it is available again.
    void WaitForOutMu(mutex_lock& lock) TF_EXCLUSIVE_LOCKS_REQUIRED(out_mu);
  };

  // Find the InstanceRec with the same instance_key as cp.  If it doesn't
  // already exist, create and initialize from gr and cp.
  //
  // Precondition: *gr must be a complete GroupRec, i.e. the value set
  // by CompleteGroupLocal. *cp must be populated with all the fields
  // required by InitInstanceSharedParams.  Ownership of InstanceRec stays
  // with this object and does not pass to the callback.
  typedef std::function<void(const Status& s, InstanceRec* ir)>
      InstanceRecCallback;
  void FindInstanceRec(const GroupRec* gr, CollectiveParams* cp,
                       const InstanceRecCallback& done)
      TF_LOCKS_EXCLUDED(instance_mu_, gr->mu, group_mu_);

  // Populate *ir with device membership from gr, then initialize to be specific
  // to cp->instance_key, i.e. order the devices and tasks.
  //
  // Preconditions:
  //  cp is populated with all DeviceLocalities
  void InitInstanceSharedParams(const GroupRec* gr, const CollectiveParams* cp,
                                InstanceRec* ir, const StatusCallback& done)
      TF_UNLOCK_FUNCTION(ir->out_mu) TF_LOCKS_EXCLUDED(gr->mu);

  void CallInitInstanceSharedParams(const GroupRec* gr,
                                    const CollectiveParams* cp, InstanceRec* ir,
                                    const InstanceRecCallback& done)
      TF_LOCKS_EXCLUDED(ir->out_mu, gr->mu);

  // Establishes the final order of ir->shared.instance.device_names and
  // ir->shared.instance.task_names by considering localities of all devices.
  void CompleteDefaultRanking(const GroupRec* gr, const CollectiveParams* cp,
                              InstanceRec* ir,
                              const std::vector<DeviceAttributes>& attributes)
      TF_EXCLUSIVE_LOCKS_REQUIRED(ir->out_mu);

  // Finish populating *cp.
  // Precondition: *gr has been fully populated by CompleteGroupLocal.
  void CompleteInstanceLocal(const string& device, const GroupRec* gr,
                             CollectiveParams* cp, bool is_source,
                             const StatusCallback& done)
      TF_LOCKS_EXCLUDED(instance_mu_, gr->mu, group_mu_);

  // Finish populating *cp from fully initialized *ir.
  // Precondition: *gr and *ir are fully populated.
  void CompleteInstanceFromInitializedIRec(const string& device,
                                           const GroupRec* gr,
                                           CollectiveParams* cp,
                                           InstanceRec* ir, bool is_source,
                                           const StatusCallback& done)
      TF_LOCKS_EXCLUDED(ir->out_mu);

  // Complete instance params after waiting for group.
  // Precondition: *cp has complete group data and default_rank.
  void WaitForGroup(InstanceRec* ir, CollectiveParams* cp, bool is_source,
                    const IRConsumer& f) TF_LOCKS_EXCLUDED(ir->out_mu);

  // If cp.device_names contains only devices local to this process
  // populates *localities, else returns an error.
  Status GetLocalDeviceLocalities(const CollectiveParams& cp,
                                  std::vector<DeviceLocality>* localities);

  // Sets CollTaskParams.is_local and CollectiveParams.default_rank.
  // Precondition: cp->device_names is fully populated and in final order.
  void CompleteTaskIsLocal(const string& task_name, CollectiveParams* cp);

  // Sets cp->instance_default_rank according to location of device in
  // current ordering of cp->instance.device_names.
  void SetDefaultRank(const string& device, CollectiveParams* cp);

  // Sets cp->instance.type based on collective op type, and attempts to assign
  // best implementation.
  void AssignCollectiveType(CollectiveParams* cp);

  // Helper to grab status under lock, invoke callback out of lock.
  void CallbackWithStatus(const InstanceRecCallback& done, InstanceRec* irec)
      TF_LOCKS_EXCLUDED(irec->out_mu);

  const bool nccl_;
  const DeviceMgr* dev_mgr_;
  DeviceResolverInterface* dev_resolver_;  // Not owned.
  string task_name_;
  mutex group_mu_;
  gtl::FlatMap<int32, std::unique_ptr<GroupRec>> group_table_
      TF_GUARDED_BY(group_mu_);
  mutex instance_mu_;
  gtl::FlatMap<int32, std::unique_ptr<InstanceRec>> instance_table_
      TF_GUARDED_BY(instance_mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_PARAM_RESOLVER_LOCAL_H_
