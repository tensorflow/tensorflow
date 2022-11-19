/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/rendezvous_util.h"
#include "tensorflow/core/platform/mutex.h"

#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {

Status SendTensorsToRendezvous(
    RendezvousInterface* rendezvous, DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    const std::vector<string>& keys, gtl::ArraySlice<Tensor> tensors_to_send) {
  if (keys.size() != tensors_to_send.size()) {
    return errors::InvalidArgument(
        "keys and tensors_to_send are not the same size. keys.size() = ",
        keys.size(), "; tensors_to_send.size() = ", tensors_to_send.size());
  }
  if (!alloc_attrs.empty() && (keys.size() != alloc_attrs.size())) {
    return errors::InvalidArgument(
        "keys and alloc_attrs are not the same size. ",
        "keys.size() = ", keys.size(),
        "; alloc_attrs.size() = ", alloc_attrs.size());
  }

  if (!rendezvous) {
    return errors::InvalidArgument("Rendezvous is null.");
  }

  Rendezvous::ParsedKey parsed;
  for (int i = 0; i < keys.size(); ++i) {
    Rendezvous::Args rendez_args;
    rendez_args.device_context = device_context;
    if (!alloc_attrs.empty()) {
      rendez_args.alloc_attrs = alloc_attrs[i];
    }
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(keys[i], &parsed));
    TF_RETURN_IF_ERROR(
        rendezvous->Send(parsed, rendez_args, tensors_to_send[i], false));
  }
  return Status::OK();
}

void RecvOutputsFromRendezvousAsync(
    RendezvousInterface* rendezvous, DeviceContext* device_context,
    const std::vector<AllocatorAttributes>& alloc_attrs,
    const std::vector<string>& keys, std::vector<Tensor>* received_tensors,
    StatusCallback done, const bool batch_allocate_tensor) {
  if (keys.empty()) {
    done(Status::OK());
    return;
  }
  if (!alloc_attrs.empty() && (keys.size() != alloc_attrs.size())) {
    done(errors::InvalidArgument(
        "keys and alloc_attrs are not the same size. ", "keys.size() = ",
        keys.size(), "; alloc_attrs.size() = ", alloc_attrs.size()));
  }

  received_tensors->reserve(keys.size());
  std::vector<
      std::tuple<string, Tensor*, Rendezvous::ParsedKey, AllocatorAttributes>>
      arguments;
  for (int i = 0; i < keys.size(); ++i) {
    Rendezvous::ParsedKey parsed;
    Status s = Rendezvous::ParseKey(keys[i], &parsed);
    received_tensors->push_back(Tensor());
    if (!s.ok()) {
      done(s);
      return;
    }
    AllocatorAttributes alloc_attr;
    if (!alloc_attrs.empty()) {
      alloc_attr = alloc_attrs[i];
    }
    arguments.emplace_back(keys[i], &((*received_tensors)[i]), parsed,
                           alloc_attr);
  }

  std::vector<Tensor*> outputs(keys.size(), nullptr);
  if (batch_allocate_tensor) {
    for (int i = 0; i < keys.size(); ++i) {
      Rendezvous::ParsedKey parsed;
      Rendezvous::ParseKey(keys[i], &parsed);
      Device* dev = rendezvous->GetRecvDeviceByParsedKey(&parsed);
      Allocator* out_allocator = dev->GetAllocator(alloc_attrs[i]);
      Tensor* in = rendezvous->GetSendTensor(&parsed);
      Tensor* out = nullptr;
      if (in && in->IsInitialized()) {
        if (in->dtype() != DT_VARIANT) {
          out = new Tensor(out_allocator, in->dtype(), in->shape(), true);
          VLOG(2) << "create batch out:" << out->DeviceSafeDebugString()
                  << ",in:" << in->DeviceSafeDebugString()
                  << ",key:" << parsed.FullKey();
        }
      }
      outputs[i] = out;
    }
    auto* gpu_device_ctx = rendezvous->GetGPUDeviceContext();
    VLOG(1) << "Batch create and single wait compute stream :"
            << gpu_device_ctx;
    auto h2d_stream = gpu_device_ctx->host_to_device_stream();
    auto compute_stream = gpu_device_ctx->stream();
    h2d_stream->ThenWaitFor(compute_stream);
  }

  auto status_cb = new ReffedStatusCallback(std::move(done));
  for (auto& p : arguments) {
    const string& key = std::get<0>(p);
    Tensor* val = std::get<1>(p);
    Rendezvous::ParsedKey parsed = std::get<2>(p);
    Rendezvous::Args rendez_args;
    rendez_args.device_context = device_context;
    rendez_args.alloc_attrs = std::get<3>(p);
    rendez_args.output = outputs[i];
    status_cb->Ref();
    rendezvous->RecvAsync(
        parsed, rendez_args,
        [val, key, status_cb](const Status& s,
                              const Rendezvous::Args& send_args,
                              const Rendezvous::Args& recv_args,
                              const Tensor& v, const bool is_dead) {
          Status status = s;
          if (status.ok()) {
            *val = v;
            if (is_dead) {
              status = errors::InvalidArgument("The tensor returned for ", key,
                                               " was not valid.");
            }
          }
          status_cb->UpdateStatus(status);
          status_cb->Unref();
        });
  }
  status_cb->Unref();
}

Status RecvOutputsFromRendezvous(RendezvousInterface* rendezvous,
                                 NamedTensors* out,
                                 const Rendezvous::Args& args) {
  // Receives values requested by the caller.
  Rendezvous::ParsedKey parsed;
  for (auto& p : *out) {
    const string& key = p.first;
    Tensor* val = &p.second;
    bool is_dead = false;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(key, &parsed));
    TF_RETURN_IF_ERROR(rendezvous->Recv(parsed, args, val, &is_dead));
    if (is_dead) {
      return errors::InvalidArgument("The tensor returned for ", key,
                                     " was not valid.");
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
