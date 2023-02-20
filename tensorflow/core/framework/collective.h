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
#ifndef TENSORFLOW_CORE_FRAMEWORK_COLLECTIVE_H_
#define TENSORFLOW_CORE_FRAMEWORK_COLLECTIVE_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/intrusive_ptr.h"

namespace tensorflow {

class BufRendezvous;
class CompleteGroupRequest;
class CompleteGroupResponse;
class CompleteInstanceRequest;
class CompleteInstanceResponse;
class Device;
class DeviceMgr;
class GetStepSequenceRequest;
class GetStepSequenceResponse;
class NcclManager;
class Tensor;

// Types of supported collective operations.
enum CollectiveType {
  REDUCTION_COLLECTIVE = 0,
  BROADCAST_COLLECTIVE,
  GATHER_COLLECTIVE,
  PERMUTE_COLLECTIVE,
  ALL_TO_ALL_COLLECTIVE,
  REDUCE_SCATTER_COLLECTIVE,
  UNDEFINED_COLLECTIVE,
};

// Some collective op implementations require runtime group configuration from
// the OpKernel.  Currently, this struct is used to set communicator key for
// NCCL-based collective implementation.
struct CollGroupRuntimeDetails {
  string communicator_key;  // for communicator-based techniques e.g. NCCL
  string ToString() const;
};

struct CollGroupMember {
  DeviceAttributes device;
  string task;
  bool is_local;
  // User provided rank
  int32 rank = -1;
};

// Data common to all members of a device group.
// All members share the same device set but its order is
// particular to an instance so it is stored there.
struct CollGroupParams {
  // Inputs from Collective ops:
  int32 group_key;
  int32 group_size;
  DeviceType device_type;
  int user_specified_rank = -1;  // rank provided by the user.
  // Generated from Collective Group Resolver:
  // Members in this group, in default rank order.
  std::vector<CollGroupMember> members;
  // True if every task has the same number of devices.
  bool same_num_devices_per_task = false;
  // Task -> number of devices on that task.
  std::unordered_map<string, int32> num_devices_per_task;
  int32 num_tasks;  // number of distinct tasks in group
  CollGroupRuntimeDetails runtime_details;
  string ToString() const;
  CollGroupParams()
      : group_key(0), group_size(0), device_type(DEVICE_CPU), num_tasks(0) {}
};

// The best implementation of a collective op depends on many factors
// including the number of devices involved, the topology of
// interconnects between them and the sizes of inputs.  This structure
// is used in generating and representing data movement choreography
// for each specific algorithm, hence it does not have a single, fixed
// interpretation.  On first execution the runtime will update this
// structure with decisions that will guide all subsequent executions.
struct CollImplDetails {
  string collective_name;
  std::vector<std::vector<int>> subdiv_permutations;
  // subdiv_offsets and max_subdivs_per_device are used together as follows:
  // When subdiv_offsets is provided (non-empty) it is used as is. When
  // subdiv_offsets is not provided subdivisons are generated dynamically
  // constrained by max_subdivs_per_device. When subdiv_offsets is empty AND
  // max_subdivs_per_device = 0 an internal default kMaxSubdivsPerDeviceDefault
  // is used. When max_subdivs_per_device = -1, no subivision is done.
  int max_subdivs_per_device = -1;  // Upper bound on subdivisions per device.
  std::vector<int> subdiv_offsets;
  std::vector<int> subdiv_source_rank;  // rank of source in each subdiv
  std::vector<int32>
      dependencies;           // collective instances on which this node depends
  string communication_hint;  // user-supplied hint for implementation choice,
                              // e.g. ring or nccl
  float timeout_seconds;      // If non zero, set a completion timeout for the
                              // collective op to detect staleness.
};

// Data common to all members of a collective instance.
// TODO(b/163171014) Refactor this struct to not be a union of all fields.
struct CollInstanceParams {
  // Identifies all participating graph nodes.
  int32 instance_key = -1;
  // The full identifier includes both instance_key and step_id.
  int64_t step_id = 0;
  CollectiveType type = UNDEFINED_COLLECTIVE;
  DataType data_type = DT_FLOAT;
  TensorShape shape = {0};
  CollImplDetails impl_details;
  string ToString() const;
  CollInstanceParams& operator=(const struct CollInstanceParams& other);
  std::vector<string> devices;  // permuter only

  // For permuter only
  // Each rank in the permutation is a receiver.
  // Indices of each rank means a sender to that rank.
  // Example: permutation = {2,0,1} means
  //   rank 0 sends to rank 2
  //   rank 1 sends to rank 0
  //   rank 2 sends to rank 1
  std::vector<int> permutation;
};

// Unique to a single CollectiveOp node.
struct CollectiveParams : public core::RefCounted {
  CollGroupParams group;
  CollInstanceParams instance;

  string name = "";        // node name used only for log or error messages
  int default_rank = -1;   // index of this op within device_names
  bool is_source = false;  // broadcast only
  int source_rank = -1;    // broadcast only
  // Rank of this device in each subdivision permutation.
  std::vector<int> subdiv_rank;
  OpKernel* merge_op = nullptr;  // reduction only
  OpKernel* final_op = nullptr;  // reduction only
  string ToString() const;
  bool run_group_initialization = true;
};

class CollectiveExecutor;

// Interface that provides resolution of device localities.
class DeviceResolverInterface {
 public:
  virtual ~DeviceResolverInterface() {}

  // Populates *attributes with the DeviceAttributes of the specified device.
  virtual Status GetDeviceAttributes(const string& device,
                                     DeviceAttributes* attributes) = 0;

  // Returns all device attributes of a task.
  virtual Status GetAllDeviceAttributes(
      const string& task, std::vector<DeviceAttributes>* attributes) = 0;

  // Updates device attributes. It returns error if any device already
  // exists in the DeviceResolver and has a different incarnation.
  virtual Status UpdateDeviceAttributes(
      const std::vector<DeviceAttributes>& attributes) = 0;
};

// Interface that provides resolution of shared CollectiveParams fields.
class ParamResolverInterface {
 public:
  virtual ~ParamResolverInterface() {}

  // Called by each collective op at first execution in order to fill out
  // the CollectiveParams structure with data gathered from the full
  // (maybe distributed) collection of peer nodes.
  virtual void CompleteParamsAsync(const DeviceAttributes& device,
                                   CollectiveParams* cp,
                                   CancellationManager* cancel_mgr,
                                   const StatusCallback& done) = 0;

  // Completes group_params with data gathered from all devices in the group.
  // This blocks until all devices are there.
  virtual void CompleteGroupAsync(const DeviceAttributes& device,
                                  CollGroupParams* group_params,
                                  CancellationManager* cancel_mgr,
                                  const StatusCallback& done) = 0;

  // Used within a distributed implementation to discover/verify data
  // shared across an instance group.
  // Note: this works differently from CompleteGroupAsync as a refactor is in
  // progress.
  virtual void CompleteInstanceAsync(const CompleteInstanceRequest* request,
                                     CompleteInstanceResponse* response,
                                     CancellationManager* cancel_mgr,
                                     const StatusCallback& done) = 0;

  // Looks up a group. It returns an error if the group is not ready or not
  // found.
  virtual Status LookupGroup(int32_t group_key, CollGroupParams* group) = 0;

  // Aborts the resolver. After abortion the resolver can no longer be used.
  virtual void StartAbort(const Status& s) = 0;
};

// Graphs which utilize Collective Ops in a common instance must
// execute with identical step_ids even if they are disjoint graphs
// run by otherwise independent tasks.  This interface supplies
// coordinated step_ids to use in such cases.
class StepSequenceInterface {
 public:
  virtual ~StepSequenceInterface() {}

  // Used with a distributed implementation to coordinate step_id
  // sequences across tasks.
  virtual void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                                    GetStepSequenceResponse* response,
                                    const StatusCallback& done) = 0;

  // Refresh the local per-graph_key step_id sequence from collective
  // group leader, if applicable.
  virtual void RefreshStepIdSequenceAsync(int64_t graph_key,
                                          const StatusCallback& done) = 0;

  // Returns the step_id that should be used for initiating a new execution
  // on the specified graph. May return the same step_id multiple times if
  // RetireStepId or RefreshStepIdReservation is not called.
  virtual int64_t NextStepId(int64_t graph_key) = 0;

  // Reports that execution of the given step has completed successfully.
  // Should be called immediately after a step completes with OK status,
  // prior to calling NextStepId().  If the step fails, don't call.
  virtual void RetireStepId(int64_t graph_key, int64_t step_id) = 0;
};

class NcclCommunicatorInterface;

// Interface that provides access to per-step CollectiveExecutor
// instances and various distributed resolution capabilities.
class CollectiveExecutorMgrInterface : public StepSequenceInterface {
 public:
  virtual ~CollectiveExecutorMgrInterface() {}

  // Returns the step-specific CollectiveExecutor, creating if one does not
  // already exist.  The caller assumes ownership of one Ref on the object.
  virtual CollectiveExecutor* FindOrCreate(int64_t step_id) = 0;

  // If there is a CollectiveExecutor for step_id, remove it from the
  // table.
  virtual void Cleanup(int64_t step_id) = 0;

  virtual ParamResolverInterface* GetParamResolver() const = 0;

  virtual DeviceResolverInterface* GetDeviceResolver() const = 0;

  virtual NcclCommunicatorInterface* GetNcclCommunicator() const = 0;
};

// Interface that a Collective Op implementation uses to exchange data
// with peers.  Note that data exchange is currently limited to types
// for which DMAHelper::CanUseDMA() returns true, i.e.  dense numeric
// types.
class CollectiveRemoteAccess {
 public:
  virtual ~CollectiveRemoteAccess() {}

  virtual void RecvFromPeer(const string& peer_device, const string& peer_task,
                            bool peer_is_local, const string& key,
                            Device* to_device, DeviceContext* to_device_ctx,
                            const AllocatorAttributes& to_alloc_attr,
                            Tensor* to_tensor,
                            const DeviceLocality& client_locality,
                            int dev_to_dev_stream_index,
                            CancellationManager* cancellation_manager,
                            const StatusCallback& done) = 0;

  virtual void PostToPeer(const string& peer_device, const string& peer_task,
                          const string& key, Device* from_device,
                          DeviceContext* from_device_ctx,
                          const AllocatorAttributes& from_alloc_attr,
                          const Tensor* from_tensor,
                          const DeviceLocality& client_locality,
                          CancellationManager* cancellation_manager,
                          const StatusCallback& done) = 0;

  // Checks the health of a collective peer. It probes the peer to see if it is
  // alive. Note that if a peer has restarted, it's considered a different one,
  // so CheckPeerHealth fails.
  virtual void CheckPeerHealth(const string& peer_task, int64_t timeout_in_ms,
                               const StatusCallback& done) = 0;

  virtual BufRendezvous* buf_rendezvous() = 0;

  virtual void StartAbort(const Status& s) = 0;
};

// A step-specific object that can execute a collective operation completely
// described by a CollectiveParams object.
class CollectiveExecutor : public core::RefCounted {
 public:
  virtual void StartAbort(const Status& s) {}

  virtual void ExecuteAsync(OpKernelContext* ctx,
                            const CollectiveParams* col_params,
                            const string& exec_key, StatusCallback done) {
    done(errors::Internal(
        "A collective Op has been called in a context in which "
        "a CollectiveExecutor has not been provided."));
  }

  virtual void CompleteParamsAsync(const DeviceAttributes& device,
                                   CollectiveParams* cp,
                                   CancellationManager* cancel_mgr,
                                   StatusCallback done) {
    done(errors::Internal(
        "A collective Op has been called in a context in which "
        "a CollectiveExecutor has not been provided."));
  }

  virtual void CompleteGroupAsync(const DeviceAttributes& device,
                                  CollGroupParams* group_params,
                                  CancellationManager* cancel_mgr,
                                  StatusCallback done) {
    return cem_->GetParamResolver()->CompleteGroupAsync(device, group_params,
                                                        cancel_mgr, done);
  }

  virtual Status LookupGroup(int32_t group_key, CollGroupParams* group) {
    return cem_->GetParamResolver()->LookupGroup(group_key, group);
  }

  // Runs the potentially-blocking closure/expensive callback.
  virtual void RunClosure(std::function<void()> closure) = 0;

  virtual CollectiveRemoteAccess* remote_access() { return nullptr; }

  // `WaitForDependencies` and `Launched` are used for fine-grained control of
  // execution order between collective instances.  These functions are intended
  // to be called in `Run` function of collective implementations, and may be
  // used to make part, or whole, of the collective execution ordered with
  // respect to other collective instances.
  //
  // `WaitForDependencies` will block until it is safe to continue the callee's
  // execution, where safety is defined as: ordered with respect to the
  // collective instances defined in the callee's `wait_for` attribute.
  virtual void WaitForDependencies(const CollectiveParams& col_params) {}
  // `UnblockDependencies` unblocks the dependent collective instances by
  // recording that this caller's device has completed the critical portion of
  // the collective execution.
  virtual void UnblockDependencies(const CollectiveParams& col_params) {}

  // Used to designate an invalid group or instance key.
  static int64_t kInvalidId;

  // Lexically scoped handle for Ref.
  class Handle {
   public:
    explicit Handle(CollectiveExecutor* ce, bool inherit_ref) : ce_(ce) {
      if (!inherit_ref) ce->Ref();
    }
    ~Handle() { ce_->Unref(); }
    CollectiveExecutor* get() const { return ce_; }

   private:
    CollectiveExecutor* ce_;
  };

 protected:
  explicit CollectiveExecutor(CollectiveExecutorMgrInterface* cem)
      : cem_(cem) {}

  // For use only by derived classes
  static OpKernelContext::Params* CtxParams(OpKernelContext* ctx);
  CollectiveExecutorMgrInterface* cem_;

  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveExecutor);
};

struct CollectiveContext {
  CollectiveExecutor* col_exec;                  // Not owned
  NcclCommunicatorInterface* nccl_communicator;  // Not owned
  const DeviceMgr* dev_mgr;                      // Not owned
  OpKernelContext* op_ctx;                       // Not owned
  OpKernelContext::Params* op_params;            // Not owned
  core::IntrusivePtr<const CollectiveParams> col_params;
  const string exec_key;
  const int64_t step_id;
  const Tensor* input;  // Not owned
  Tensor* output;       // Not owned
  Device* device;       // The device for which this instance labors
  const string device_name;
  DeviceLocality device_locality;

  CollectiveContext(CollectiveExecutor* col_exec,
                    NcclCommunicatorInterface* nccl_communicator,
                    const DeviceMgr* dev_mgr, OpKernelContext* ctx,
                    OpKernelContext::Params* op_params,
                    const CollectiveParams* col_params, const string& exec_key,
                    int64_t step_id, const Tensor* input, Tensor* output);
};

class NcclCommunicatorInterface {
 public:
  virtual ~NcclCommunicatorInterface() = default;

  virtual string GenerateCommunicatorKey() = 0;

  virtual void Enqueue(std::shared_ptr<CollectiveContext> col_ctx,
                       StatusCallback done) = 0;

  virtual void StartAbort(const Status& s) = 0;
};

// Interface of a Collective Op implementation.  Each specific CollectiveOp will
// implement this interface and register the implementation via the
// CollectiveRegistry detailed below.  See common_runtime/ring_reducer and
// common_runtime/hierarchical_tree_broadcaster for examples.
class CollectiveImplementationInterface : public core::RefCounted {
 public:
  virtual ~CollectiveImplementationInterface() = default;

  // Initializes the portions of `col_params` specific to this
  // implementation.  Called exactly once for every Collective instance during
  // the CollectiveParams resolution process when the graph is first executed,
  // at the end of `CompleteInstanceLocal()`.
  // NOTE(ayushd): This is effectively a static function because it modifies the
  // `col_params` passed in and should not manipulate any data members.  However
  // because it is virtual and needs to be implemented by every derived class we
  // do not mark it as static.
  virtual Status InitializeCollectiveParams(CollectiveParams* col_params) = 0;

  // Prepares the CollectiveContext for executing this CollectiveImplementation.
  // Called from CollectiveExecutor right before calling Run().  The
  // CollectiveContext passed in must outlive the CollectiveImplementation
  // object.
  virtual Status InitializeCollectiveContext(
      std::shared_ptr<CollectiveContext> col_ctx) = 0;

  // Processes and moves data according to the logic of this Collective
  // implementation.  Relies on appropriate initialization of op-specific
  // CollectiveParams in InitializeCollectiveParams(), as well as appropriate
  // context initialization in InitializeCollectiveContext().
  virtual void Run(StatusCallback done) = 0;
};

// Static-methods only class for registering and looking up collective
// implementations.
class CollectiveRegistry {
 public:
  using Factory = std::function<CollectiveImplementationInterface*()>;
  // Looks up a previously registered CollectiveImplementation under
  // `collective_name`.  If found, creates an instance of the implementation and
  // assign to `implementation`.
  static Status Lookup(const string& collective_name,
                       CollectiveImplementationInterface** implementation);

  // Looks up a previously registered CollectiveImplementation under
  // `collective_name`.  If found, returns the static instance of this
  // implementation via `implementation`.  This instance should only be used to
  // call InitializateCollectiveParams.
  static Status LookupParamResolverInstance(
      const string& collective_name,
      CollectiveImplementationInterface** implementation);

  // Returns all registered collective implementations.
  static void GetAll(
      std::vector<CollectiveImplementationInterface*>* implementations);

 private:
  friend class CollectiveRegistration;
  // Registers a CollectiveImplementation with name `collective_name` and
  // factory `factory`.  The latter is a function used to create instances of
  // the CollectiveImplementation.  Also creates a static instance of the
  // implementation - this instance is used during param resolution and should
  // only be used to call InitializeCollectiveParams.
  static Status Register(const string& collective_name, Factory factory);

  static Status LookupHelper(const string& collective_name,
                             CollectiveImplementationInterface** implementation,
                             bool param_resolver);
};

// Class used to call CollectiveRegistry::Register.  This should only be used to
// create a global static object.
class CollectiveRegistration {
 public:
  CollectiveRegistration(const string& collective_name,
                         CollectiveRegistry::Factory factory) {
    TF_CHECK_OK(CollectiveRegistry::Register(collective_name, factory));
  }
};

#define REGISTER_COLLECTIVE(name, implementation)             \
  static CollectiveRegistration register_##name##_collective( \
      #name, []() { return new implementation; });

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_COLLECTIVE_H_
