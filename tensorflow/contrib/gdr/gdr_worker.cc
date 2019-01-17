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

#include "tensorflow/contrib/gdr/gdr_worker.h"

#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

GdrWorker::GdrWorker(WorkerEnv* worker_env, const ConfigProto& config,
                     RemoteMemoryManager* remote_memory_manager)
    : GrpcWorker(worker_env, config),
      remote_memory_manager_(remote_memory_manager),
      recent_request_ids_(100000) {}

void GdrWorker::GrpcRecvTensorAsync(CallOptions* opts,
                                    const RecvTensorRequest* request,
                                    ::grpc::ByteBuffer* response,
                                    StatusCallback done) {
  Status s = recent_request_ids_.TrackUnique(
      request->request_id(), "RecvTensor (GdrWorker)", *request);
  if (!s.ok()) {
    done(s);
    return;
  }

  const int64 step_id = request->step_id();
  const string& key = request->rendezvous_key();
  TRACEPRINTF("RecvTensor: %lld %s", step_id, key.c_str());
  Rendezvous::ParsedKey parsed;
  s = Rendezvous::ParseKey(key, &parsed);
  Device* src_dev = nullptr;
  if (s.ok()) {
    s = PrepareRecvTensor(parsed, &src_dev);
  }
  if (!s.ok()) {
    done(s);
    return;
  }

  // Request the tensor associated with the rendezvous key. Any time
  // while waiting for the tensor to be produced, up until the start
  // of execution of the callback lambda body below, an RPC
  // cancellation should abort the rendezvous.
  opts->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
  const bool dma_ok = request->dma_ok();
  env_->rendezvous_mgr->RecvLocalAsync(
      step_id, parsed,
      [this, opts, response, done, src_dev, request, dma_ok](
          const Status& status, const Rendezvous::Args& send_args,
          const Rendezvous::Args&, const Tensor& val, const bool is_dead) {
        opts->ClearCancelCallback();
        if (status.ok()) {
          // DMA can only be used for Tensors that do not fall into
          // the following three odd edge cases: 1) a zero-size
          // buffer, 2) a dead tensor which has an uninit value, and
          // 3) the tensor has the on_host allocation attribute,
          // i.e. it's in CPU RAM *independent of its assigned
          // device type*.
          const bool on_host = send_args.alloc_attrs.on_host();
          if (val.TotalBytes() > 1024 && (!is_dead) &&
              DMAHelper::CanUseDMA(&val) && dma_ok) {
            // DMA cases.
            RecvTensorResponse* proto = new RecvTensorResponse;
            proto->set_is_dead(is_dead);
            proto->set_send_start_micros(Env::Default()->NowMicros());
            TensorProto* tensor_proto = proto->mutable_tensor();
            tensor_proto->set_dtype(val.dtype());
            val.shape().AsProto(tensor_proto->mutable_tensor_shape());
            auto transport_options = proto->mutable_transport_options();
            remote_memory_manager_->TransportOptionsFromTensor(
                transport_options, val, src_dev, send_args.device_context,
                on_host, [proto, done, response](const Status& s) {
                  if (s.ok()) {
                    grpc::EncodeRecvTensorResponseToByteBuffer(*proto,
                                                               response);
                    done(Status::OK());
                  } else {
                    done(s);
                  }
                  delete proto;
                });
          } else {
            // Non-DMA cases.
            if (src_dev->tensorflow_gpu_device_info() && (!on_host)) {
              DeviceContext* send_dev_context = send_args.device_context;
              AllocatorAttributes alloc_attrs;
              alloc_attrs.set_gpu_compatible(true);
              alloc_attrs.set_on_host(true);
              Allocator* alloc = src_dev->GetAllocator(alloc_attrs);
              Tensor* copy = new Tensor(alloc, val.dtype(), val.shape());
              CHECK(send_dev_context)
                  << "send dev name: " << src_dev->name()
                  << " gpu_info: " << src_dev->tensorflow_gpu_device_info();
              // "val" is on an accelerator device. Uses the device_context to
              // fill the copy on host.
              StatusCallback copy_ready = [response, done, copy,
                                           is_dead](const Status& s) {
                // The value is now ready to be returned on the wire.
                grpc::EncodeTensorToByteBuffer(is_dead, *copy, response);
                done(s);
                delete copy;
              };

              send_dev_context->CopyDeviceTensorToCPU(
                  &val, request->rendezvous_key(), src_dev, copy, copy_ready);
            } else {
              grpc::EncodeTensorToByteBuffer(is_dead, val, response);
              done(Status::OK());
            }
          }
        } else {
          //  !s.ok()
          done(status);
        }
      });
}

void GdrWorker::RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                             RecvBufResponse* response, StatusCallback done) {
  // This is an RDMA enabled implementation augmenting grpc.
  Status s = recent_request_ids_.TrackUnique(
      request->request_id(), "RecvBuf (GdrWorker)", *request);
  if (!s.ok()) {
    done(s);
    return;
  }
  CollectiveExecutor::Handle ce_handle(
      env_->collective_executor_mgr->FindOrCreate(request->step_id()), true);
  CollectiveRemoteAccess* rma = ce_handle.get()->remote_access();
  rma->buf_rendezvous()->ConsumeBuf(
      request->buf_rendezvous_key(),
      [this, request, response, done](const Status& status,
                                      BufRendezvous::Hook* hook) {
        Status s = status;
        if (s.ok()) {
          if (!DMAHelper::CanUseDMA(hook->prod_value)) {
            s = errors::Internal("Tensor value for key ",
                                 request->buf_rendezvous_key(),
                                 " is not of a type supported by RecvBuf");
          }
        }
        if (s.ok()) {
          remote_memory_manager_->TransportOptionsFromTensor(
              response->mutable_transport_options(), *hook->prod_value,
              hook->prod_dev, hook->prod_ctx, hook->prod_attr.on_host(),
              [this, response, done, hook](const Status& s) {
                response->set_send_start_micros(env_->env->NowMicros());
                done(s);
                BufRendezvous::DoneWithHook(hook);
              });
        }
      });
}

}  // namespace tensorflow
