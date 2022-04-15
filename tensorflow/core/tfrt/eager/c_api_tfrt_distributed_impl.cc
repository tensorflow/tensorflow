/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/eager/c_api_tfrt_distributed_impl.h"

#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tfrt/distributed_runtime/distributed_context.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/distributed_init_helper.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/fabric_communicator.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/proto/cluster_config.pb.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/server_context.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/task_name_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

constexpr char kRemotePrefix[] = "remote_";

std::string GetTensorFlowDeviceType(string_view name) {
  int pos = name.find(kRemotePrefix);
  return absl::AsciiStrToUpper(
      pos == 0 ? name.substr(strlen(kRemotePrefix)).str() : name.str());
}

DistributedContextConfiguration ConvertServerDefToDistributedConfiguration(
    const tensorflow::ServerDef& server_def) {
  DistributedContextConfiguration dist_config;
  dist_config.set_job_name(server_def.job_name());
  dist_config.set_task_id(server_def.task_index());
  ClusterConfiguration* cluster_config = dist_config.mutable_cluster_config();
  // Currently take the first task in the first job as collective group leader.
  // TODO(haoyuzhang): Make this configurable from API by reading from
  // `config.experimental.collective_group_leader`.
  cluster_config->set_lead_task_name(TaskNameUtil::ConcatTaskName(
      server_def.cluster().job(0).name(), /*task_id=*/0));
  for (const auto& job_def : server_def.cluster().job()) {
    JobConfiguration* job_config = cluster_config->add_jobs();
    job_config->set_name(job_def.name());
    *job_config->mutable_tasks() = job_def.tasks();
  }
  return dist_config;
}

std::unique_ptr<ServerContext> CreateServer(
    const DistributedContextConfiguration& dist_config, HostContext* host_ctx) {
  const std::string& job_name = dist_config.job_name();
  const int task_id = dist_config.task_id();
  std::string server_address;
  for (const auto& job_config : dist_config.cluster_config().jobs()) {
    if (job_config.name() == job_name) {
      server_address = job_config.tasks().at(task_id);
      break;
    }
  }
  FabricCommunicatorConfiguration fabric_config{"grpc_communicator",
                                                server_address};
  ServerContextConfiguration server_config{fabric_config};
  return std::make_unique<ServerContext>(host_ctx, server_config);
}

}  // namespace

class DistributedManagerContextImpl
    : public DistributedManagerContextInterface {
 public:
  explicit DistributedManagerContextImpl(HostContext* host_context);

  tensorflow::Status SetOrUpdateServerDef(
      const tensorflow::ServerDef& server_def, bool reset_context,
      int keep_alive_secs) override;

  tensorflow::Status EnableCollectiveOps(
      const tensorflow::ServerDef& server_def) override;

  tensorflow::Status CheckRemoteAlive(const std::string& remote_task_name,
                                      bool* is_alive) override;

  tensorflow::CoordinationServiceAgent* GetCoordinationServiceAgent() override;

  void UpdateRequestContextBuilder(RequestContextBuilder* builder) override;
  void PopulateRemoteDevices(tensorflow::DeviceSet* dev_set) override;

 private:
  HostContext* host_context_;
  std::unique_ptr<tfrt::ServerContext> server_context_;
  AsyncValueRef<tfrt::DistributedContext> dist_context_;
  std::unique_ptr<tensorflow::StaticDeviceMgr> tf_devices_;
};

DistributedManagerContextImpl::DistributedManagerContextImpl(
    HostContext* host_context)
    : host_context_(host_context) {
  TaskNameUtil::SetUseReplicaInTaskName();
}

tensorflow::Status DistributedManagerContextImpl::SetOrUpdateServerDef(
    const tensorflow::ServerDef& server_def, bool reset_context,
    int keep_alive_secs) {
#if defined(PLATFORM_GOOGLE)
  DistributedContextConfiguration dist_config =
      ConvertServerDefToDistributedConfiguration(server_def);
  server_context_ = CreateServer(dist_config, host_context_);

  // Create distributed contexts on current and remote tasks. Implemented as a
  // blocking call to be consistent with the behavior of current TF.
  const DistributedInitHelper* init_helper =
      server_context_->GetDistributedInitHelper();
  absl::Notification n;
  init_helper->InitializeSingleClientDistributedContext(
      std::move(dist_config),
      [&n, this](Expected<DistributedContext*> expected) mutable {
        if (!expected) tfrt::DieIfError(expected.takeError());
        const uint64_t cid = expected.get()->GetContextId();
        dist_context_ = server_context_->GetDistributedContextAsyncValue(cid);
        n.Notify();
      });
  n.WaitForNotification();

  auto device_refs =
      dist_context_->GetRemoteDeviceManager()->ListDevices<Device>();
  std::vector<std::unique_ptr<tensorflow::Device>> tf_devices;
  for (auto& device_ref : device_refs) {
    tensorflow::DeviceAttributes da;
    da.set_name(device_ref->name().str());
    da.set_device_type(GetTensorFlowDeviceType(device_ref->type().name()));
    // TF Devices created here might not have all of the attributes needed.
    // Currently, it is only used by Placer during TFRT Function creation.
    tf_devices.emplace_back(NewRemoteDevice(tensorflow::Env::Default(), da));
  }
  tf_devices_ =
      std::make_unique<tensorflow::StaticDeviceMgr>(std::move(tf_devices));
  return tensorflow::Status::OK();
#endif  // PLATFORM_GOOGLE
  return tensorflow::errors::Unimplemented(
      "SetOrUpdateServerDef in open source is not yet implemented.");
}

tensorflow::Status DistributedManagerContextImpl::EnableCollectiveOps(
    const tensorflow::ServerDef& server_def) {
#if defined(PLATFORM_GOOGLE)
  DistributedContextConfiguration dist_config =
      ConvertServerDefToDistributedConfiguration(server_def);
  server_context_ = CreateServer(dist_config, host_context_);

  DistributedInitHelper* init_helper =
      server_context_->GetDistributedInitHelper();
  absl::Notification n;
  init_helper->InitializeMultiClientDistributedContext(
      std::move(dist_config),
      [&n, this](Expected<DistributedContext*> expected) mutable {
        if (!expected) tfrt::DieIfError(expected.takeError());
        const uint64_t cid = expected.get()->GetContextId();
        dist_context_ = server_context_->GetDistributedContextAsyncValue(cid);
        n.Notify();
      });
  n.WaitForNotification();

  return tensorflow::Status::OK();
#endif  // PLATFORM_GOOGLE
  return tensorflow::errors::Unimplemented(
      "EnableCollectiveOps in open source is not yet implemented.");
}

tensorflow::Status DistributedManagerContextImpl::CheckRemoteAlive(
    const std::string& remote_task_name, bool* is_alive) {
  return tensorflow::errors::Unimplemented(
      "CheckRemoteAlive in TFRT is not yet implemented.");
}

tensorflow::CoordinationServiceAgent*
DistributedManagerContextImpl::GetCoordinationServiceAgent() {
  TFRT_LOG(FATAL) << "Coordination service in TFRT is not yet enabled.";
  return nullptr;
}

void DistributedManagerContextImpl::UpdateRequestContextBuilder(
    RequestContextBuilder* builder) {
  builder->context_data().insert(dist_context_.CopyRef());
}

void DistributedManagerContextImpl::PopulateRemoteDevices(
    tensorflow::DeviceSet* dev_set) {
  if (tf_devices_ == nullptr) {
    return;
  }
  for (auto& device : tf_devices_->ListDevices()) {
    dev_set->AddDevice(device);
  }
}

std::unique_ptr<DistributedManagerContextInterface>
CreateDistributedManagerContext(HostContext* host_context) {
  return std::make_unique<DistributedManagerContextImpl>(host_context);
}

}  // namespace tf
}  // namespace tfrt
