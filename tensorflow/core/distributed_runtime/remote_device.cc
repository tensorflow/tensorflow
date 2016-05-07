/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/remote_device.h"

#include <vector>
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

using std::placeholders::_1;

// TODO(zhifengc): We need to consolidate (full/partial) device name
// parsing into one place.
//
// Parses and returns the local device part (e.g., cpu:0, gpu:4).
string GetLocalDeviceName(StringPiece fullname) {
  auto pos = fullname.rfind('/');
  CHECK_NE(pos, StringPiece::npos);
  fullname.remove_prefix(pos + 1);
  return fullname.ToString();
}

class RemoteDevice : public Device {
 public:
  RemoteDevice(Env* env, const DeviceAttributes& da, WorkerInterface* wi)
      : Device(env, da, nullptr),
        local_dev_name_(GetLocalDeviceName(da.name())),
        wi_(wi) {}

  ~RemoteDevice() override { delete wi_; }
  Status Sync() override { return Status::OK(); }
  Allocator* GetAllocator(AllocatorAttributes attr) override { return nullptr; }

 private:
  const string local_dev_name_;
  WorkerInterface* wi_;

  TF_DISALLOW_COPY_AND_ASSIGN(RemoteDevice);
};

void NewRemoteDevices(Env* env, WorkerCacheInterface* worker_cache,
                      const string& worker_name, NewRemoteDevicesDone done) {
  WorkerInterface* wi = worker_cache->CreateWorker(worker_name);
  if (wi == nullptr) {
    std::vector<Device*> empty;
    done(errors::NotFound("Device ", worker_name, " is not found."), &empty);
    return;
  }
  struct Call {
    GetStatusRequest req;
    GetStatusResponse resp;
  };
  Call* call = new Call;
  auto cb = [env, worker_cache, worker_name, done, wi, call](const Status& s) {
    std::vector<Device*> remote_devices;
    if (s.ok()) {
      remote_devices.reserve(call->resp.device_attributes_size());
      for (const DeviceAttributes& da : call->resp.device_attributes()) {
        auto d =
            new RemoteDevice(env, da, worker_cache->CreateWorker(worker_name));
        remote_devices.push_back(d);
      }
    }
    done(s, &remote_devices);
    delete wi;
    delete call;
  };
  wi->GetStatusAsync(&call->req, &call->resp, cb);
}

}  // namespace tensorflow
