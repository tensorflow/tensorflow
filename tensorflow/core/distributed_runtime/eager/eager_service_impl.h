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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_EAGER_SERVICE_IMPL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_EAGER_SERVICE_IMPL_H_

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/eager/remote_tensor_handle.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

// A TensorFlow Eager Worker runs ops and supports worker to worker
// Tensor transfer.
//
// See eager_service.proto for more details about each method.
// This class can be wrapped by specific classes that implement rpc transports
// over this (e.g. gRPC).
class EagerServiceImpl {
 public:
  explicit EagerServiceImpl(const WorkerEnv* env) : env_(env) {
    gc_thread_.reset(
        env_->env->StartThread({}, "EagerServiceContextGC", [this]() {
          while (true) {
            {
              mutex_lock l(gc_thread_shutdown_mu_);
              gc_thread_cv_.wait_for(l, std::chrono::seconds(1));

              if (shutting_down_) {
                return;
              }
            }
            {
              mutex_lock l(contexts_mu_);
              for (auto it = contexts_.begin(); it != contexts_.end();) {
                if (it->second->IsStale()) {
                  it->second->Unref();
                  it = contexts_.erase(it);
                } else {
                  it++;
                }
              }
            }
          }
        }));
  }
  virtual ~EagerServiceImpl() {
    {
      mutex_lock l(gc_thread_shutdown_mu_);
      shutting_down_ = true;
      gc_thread_cv_.notify_all();
    }
    gc_thread_.reset();

    mutex_lock l(contexts_mu_);
    for (auto& entry : contexts_) {
      entry.second->Unref();
    }
  }

  Status CreateContext(const CreateContextRequest* request,
                       CreateContextResponse* response);

  Status UpdateContext(const UpdateContextRequest* request,
                       UpdateContextResponse* response);

  // Create a ServerContext for master eager context.
  Status CreateMasterContext(const tensorflow::uint64 context_id,
                             EagerContext* context);

  static const uint64 kInvalidStreamId = 0;

  // Used by both Enqueue and StreamingEnqueue RPCs.
  Status Enqueue(const EnqueueRequest* request, EnqueueResponse* response,
                 uint64 stream_id = kInvalidStreamId);

  Status WaitQueueDone(const WaitQueueDoneRequest* request,
                       WaitQueueDoneResponse* response);

  Status KeepAlive(const KeepAliveRequest* request,
                   KeepAliveResponse* response);

  Status CloseContext(const CloseContextRequest* request,
                      CloseContextResponse* response);

 protected:
  // This is the server-side execution context. All state regarding execution of
  // a client's ops is held in this server-side context (all generated tensors,
  // and the EagerContext).
  class ServerContext : public core::RefCounted {
   public:
    // Create a ServerContext for local master.
    static ServerContext* CreateMasterContext(tensorflow::EagerContext* ctx,
                                              const WorkerEnv* env) {
      return new ServerContext(ctx, -1, env, /* is_master= */ true);
    }

    explicit ServerContext(tensorflow::EagerContext* ctx,
                           int64 destroy_after_secs, const WorkerEnv* env,
                           const bool is_master = false)
        : ctx_(ctx), env_(env), is_master_(is_master) {
      ctx->Ref();
      destroy_after_micros_ =
          destroy_after_secs * tensorflow::EnvTime::kSecondsToMicros;
      RecordAccess();
    }

    ~ServerContext() override {
      // TFE_Context is responsible for shutting down master eager context.
      if (!is_master_) {
        ctx_->WaitForAndCloseRemoteContexts();
      }
      // ctx_->RefCountIsOne() should be true here when is_master_ = false.
      // TODO(iga): Remove EagerContext refcounting.
      ctx_->Unref();
    }

    tensorflow::EagerContext* Context() const { return ctx_; }

    void RecordAccess() {
      mutex_lock l(last_accessed_mu_);
      last_accessed_micros_ = env_->env->NowMicros();
    }

    bool IsStale() {
      mutex_lock l(last_accessed_mu_);
      return (destroy_after_micros_ > 0 &&
              (env_->env->NowMicros() - last_accessed_micros_) >
                  destroy_after_micros_);
    }

   private:
    // The context for this execution.
    tensorflow::EagerContext* ctx_;

    const WorkerEnv* const env_;  // Not owned.

    mutex last_accessed_mu_;
    int64 last_accessed_micros_ GUARDED_BY(last_accessed_mu_);
    int64 destroy_after_micros_;

    const bool is_master_;
  };
  // The returned ServerContext will need to be Unrefed.
  tensorflow::Status GetServerContext(uint64, ServerContext**);

  class ClientTensorHandleDeleteNode : public EagerNode {
   public:
    ClientTensorHandleDeleteNode(
        ServerContext* context,
        std::unique_ptr<RemoteTensorHandleInternal> handle_to_delete)
        : tensorflow::EagerNode(),
          context_(context),
          handle_to_delete_(std::move(handle_to_delete)) {
      context_->Ref();
    }

    ~ClientTensorHandleDeleteNode() override { context_->Unref(); }

    Status Run() override {
      VLOG(3) << "ServerContext: Deleting tensor handle "
              << handle_to_delete_->op_id << ":"
              << handle_to_delete_->output_num;
      return context_->Context()->RemoteMgr()->DeleteTensorHandle(
          *handle_to_delete_);
    }

    void Abort(Status status) override {}

    string DebugString() const override {
      string out = "[ClientTensorHandleDeleteNode]";
      strings::StrAppend(&out, " op_id: ", handle_to_delete_->op_id);
      strings::StrAppend(&out, ", output_num: ", handle_to_delete_->output_num);
      return out;
    }

   private:
    // Owns one reference.
    ServerContext* const context_;
    const std::unique_ptr<RemoteTensorHandleInternal> handle_to_delete_;
  };

 private:
  Status ExecuteOp(const Operation& operation, EagerContext* eager_context,
                   EagerExecutor* eager_executor,
                   QueueResponse* queue_response);
  Status SendTensor(const SendTensorOp& send_tensor,
                    EagerContext* eager_context);
  Status RegisterFunction(const RegisterFunctionOp& register_function,
                          EagerContext* eager_context);
  Status CleanupFunction(const CleanupFunctionOp& cleanup_function);
  const WorkerEnv* const env_;  // Not owned.

  mutex contexts_mu_;
  std::unordered_map<uint64, ServerContext*> contexts_ GUARDED_BY(contexts_mu_);

  // Mutex to guard access to EagerContext in `contexts_`. Different from
  // `contexts_mu_` which guards adding / removing item from the map, this mutex
  // is supposed to be used to avoid concurrent reading/updating the state of an
  // EagerContext inside the map.
  mutex context_update_mu_;

  std::unique_ptr<Thread> gc_thread_;
  mutex gc_thread_shutdown_mu_;
  condition_variable gc_thread_cv_;
  bool shutting_down_ GUARDED_BY(gc_thread_shutdown_mu_) = false;

  TF_DISALLOW_COPY_AND_ASSIGN(EagerServiceImpl);
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_EAGER_SERVICE_IMPL_H_
