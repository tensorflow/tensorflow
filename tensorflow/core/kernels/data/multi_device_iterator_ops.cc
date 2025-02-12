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
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/metric_utils.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/data/unbounded_thread_pool.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace tensorflow {
namespace data {
namespace {

const char kAnonymousMultiDeviceIterator[] = "AnonymousMultiDeviceIterator";
const char kAnonymousMultiDeviceIteratorV3[] = "AnonymousMultiDeviceIteratorV3";
const char kDevices[] = "devices";
const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";

struct HostBufferElement {
  absl::Status status;
  bool end_of_sequence;
  std::vector<Tensor> value;
};

using MultiDeviceIteratorCallback =
    std::function<void(const HostBufferElement&)>;

// MultiDeviceIterator provides the ability for multiple devices to fetch from
// one iterator in a roundrobin sequence, which is deterministic. This means
// that, for exmaple, starting from the beginning GetNextFromShard(0) always
// gets the first element and GetNextFromShard(1) always gets the second
// element, even if GetNextFromShard(1) is called before GetNextFromShard(0).
//
// Note on cancellation:
//   * MultiDeviceIterator can be cancelled as a whole by calling Reset() or
//   cancel MultiDeviceIterator::cancellation_manager().
//   * GetNextFromShard can be cancelled independently. Cancelling
//   GetNextFromShard for one shard doesn't cancel the underlying prefetching,
//   nor does it other calls of GetNextFromShard.
class MultiDeviceIterator : public ResourceBase {
 public:
  MultiDeviceIterator(
      Env* env, const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      const std::vector<string>& devices,
      std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* flr,
      std::unique_ptr<FunctionHandleCache> function_handle_cache)
      : metrics_collector_(flr ? flr->device()->device_type() : DEVICE_DEFAULT,
                           *env),
        unbounded_thread_pool_(env, "tf_data_multi_device_iterator_resource"),
        output_types_(output_types),
        output_shapes_(output_shapes),
        devices_(devices),
        flib_def_(std::move(flib_def)),
        flr_(flr),
        pflr_(std::move(pflr)),
        function_handle_cache_(std::move(function_handle_cache)) {
    DCHECK(flr_ != nullptr);
    VLOG(2) << "Creating multi-device iterator.";
  }

  ~MultiDeviceIterator() override {
    VLOG(2) << "Destroying multi-device iterator.";
  }

  string DebugString() const override {
    return strings::StrCat("MultiDeviceIterator for ", devices_.size(),
                           " devices");
  }

  absl::Status Init(std::unique_ptr<IteratorBase> iterator,
                    int64_t max_buffer_size, int64_t* incarnation_id,
                    DatasetBase* dataset) {
    if (iterator) {
      TF_RETURN_IF_ERROR(
          VerifyTypesMatch(output_types_, iterator->output_dtypes()));
      TF_RETURN_IF_ERROR(
          VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));
    }

    mutex_lock l(mu_);
    if (multi_device_buffer_) {
      multi_device_buffer_->Reset();
    }
    dataset->Ref();
    dataset_.reset(dataset);

    ++incarnation_id_;
    *incarnation_id = incarnation_id_;

    multi_device_buffer_ = std::make_unique<MultiDeviceBuffer>(
        devices_.size(), max_buffer_size, incarnation_id_, std::move(iterator),
        this);
    return absl::OkStatus();
  }

  absl::Status GetNextFromShard(OpKernelContext* ctx, int shard_num,
                                int64_t incarnation_id,
                                MultiDeviceIteratorCallback callback) {
    tsl::profiler::TraceMe traceme([&] {
      return tsl::profiler::TraceMeEncode(
          absl::StrCat("GetNextFromShard", shard_num),
          {{"shard_num", shard_num}});
    });
    tf_shared_lock l(mu_);
    IteratorContext::Params params(ctx);
    params.flr = flr_;
    params.function_handle_cache = function_handle_cache_.get();
    params.resource_mgr = &resource_mgr_;
    params.thread_factory = unbounded_thread_pool_.get_thread_factory();
    params.thread_pool = &unbounded_thread_pool_;
    params.cancellation_manager = ctx->cancellation_manager();
    IteratorContext iter_ctx(std::move(params));
    multi_device_buffer_->GetNextFromShard(&iter_ctx, shard_num, incarnation_id,
                                           std::move(callback));
    return absl::OkStatus();
  }

  const DataTypeVector& output_types() const { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

  FunctionLibraryRuntime* const flr() {
    tf_shared_lock l(mu_);
    return flr_;
  }

  FunctionHandleCache* function_handle_cache() {
    return function_handle_cache_.get();
  }

  ResourceMgr* resource_mgr() { return &resource_mgr_; }

  CancellationManager* cancellation_manager() { return &cancellation_manager_; }

  IteratorMetricsCollector& metrics_collector() { return metrics_collector_; }

 private:
  // A private class that uses a background thread to keep a per device buffer
  // full.
  class MultiDeviceBuffer {
   public:
    MultiDeviceBuffer(size_t size, int64_t max_buffer_size,
                      int64_t incarnation_id,
                      std::unique_ptr<IteratorBase> host_iterator,
                      MultiDeviceIterator* parent)
        : buffer_(size),
          size_(size),
          max_buffer_size_(max_buffer_size),
          incarnation_id_(incarnation_id),
          host_iterator_(std::move(host_iterator)),
          parent_(parent) {}

    ~MultiDeviceBuffer() {
      {
        mutex_lock l(mu_);
        if (!background_thread_started_) return;
      }
      Reset();
    }

    void Reset() TF_LOCKS_EXCLUDED(mu_) {
      {
        mutex_lock l(mu_);
        if (background_thread_ && !background_thread_finished_) {
          cancellation_manager_.StartCancel();
          // Wake up the background thread.
          for (int i = 0; i < size_; ++i) {
            buffer_[i].cond_var.notify_all();
          }

          // Make sure background thread has finished first.
          while (!background_thread_finished_) {
            shutdown_cond_var_.wait(l);
          }
        }
      }
      RunPendingCallbacks();
    }

    void GetNextFromShard(IteratorContext* ctx, int shard_num,
                          int64_t incarnation_id,
                          MultiDeviceIteratorCallback callback) {
      HostBufferElement elem;
      if (incarnation_id_ != incarnation_id) {
        elem.status = errors::InvalidArgument(
            "Invalid incarnation id. Provided: ", incarnation_id,
            "; Expected: ", incarnation_id_);
        callback(elem);
        return;
      }

      bool produced_output = false;
      {
        mutex_lock l(mu_);
        if (cancellation_manager_.IsCancelled()) {
          elem.status = errors::Cancelled("Cancelled Multidevice iterator");
          callback(elem);
          return;
        }

        EnsureBackgroundThreadStarted(ctx);

        if (!buffer_[shard_num].data.empty()) {
          produced_output = true;
          std::swap(elem, buffer_[shard_num].data.front());
          buffer_[shard_num].data.pop_front();
          // Wake up background thread if it is blocked on this element.
          if (buffer_[shard_num].data.size() == max_buffer_size_ - 1) {
            buffer_[shard_num].cond_var.notify_all();
          }
        } else {
          if (end_of_iterator_) {
            produced_output = true;
            elem.end_of_sequence = true;
          } else {
            auto callback_container =
                std::make_shared<HostBuffer::CallbackContainer>(
                    std::move(callback));
            elem.status = RegisterCancellationCallback(
                ctx->cancellation_manager(),
                [callback_container]() {
                  if (callback_container->is_called.exchange(true)) {
                    return;
                  }
                  HostBufferElement elem;
                  elem.status =
                      errors::Cancelled("GetNextFromShard was cancelled");
                  callback_container->callback(elem);
                },
                &callback_container->deregister_cancellation);
            if (!elem.status.ok()) {
              callback_container->callback(elem);
              return;
            }
            buffer_[shard_num].callbacks.push_back(
                std::move(callback_container));
            buffer_[shard_num].cond_var.notify_all();
            callback = nullptr;
          }
        }
      }

      if (produced_output) {
        callback(elem);
      }
    }

   private:
    void EnsureBackgroundThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!background_thread_) {
        IteratorContext::Params params(ctx);
        params.cancellation_manager = &cancellation_manager_;
        background_thread_ =
            parent_->unbounded_thread_pool_.get_thread_factory()->StartThread(
                "tf_data_multi_device_iterator",
                std::bind(
                    &MultiDeviceIterator::MultiDeviceBuffer::BackgroundThread,
                    this,
                    std::make_shared<IteratorContext>(std::move(params))));
      }
    }

    void RunPendingCallbacks() TF_LOCKS_EXCLUDED(mu_) {
      // Run all remaining callbacks.

      std::vector<std::shared_ptr<HostBuffer::CallbackContainer>>
          callback_containers;
      std::vector<HostBufferElement> cancellation_elements;
      {
        mutex_lock l(mu_);

        for (int i = 0; i < size_; ++i) {
          while (!buffer_[i].callbacks.empty()) {
            if (buffer_[i].callbacks.front()->is_called.exchange(true)) {
              buffer_[i].callbacks.pop_front();
              continue;
            }
            if (buffer_[i].data.empty()) {
              HostBufferElement elem;
              if (end_of_iterator_) {
                elem.end_of_sequence = true;
              } else {
                elem.status =
                    errors::Cancelled("Cancelled and buffer not filled.");
              }
              cancellation_elements.push_back(std::move(elem));
            } else {
              cancellation_elements.push_back(
                  std::move(buffer_[i].data.front()));
              buffer_[i].data.pop_front();
            }
            callback_containers.push_back(
                std::move(buffer_[i].callbacks.front()));
            buffer_[i].callbacks.pop_front();
          }
        }
      }
      for (int i = 0; i < callback_containers.size(); ++i) {
        if (callback_containers[i]->deregister_cancellation != nullptr) {
          callback_containers[i]->deregister_cancellation();
        }
        // We invoke the callback regardless of whether deregistration succeeds
        // or not, because we have set is_called=true previous which effectively
        // disables the cancellation callback.
        callback_containers[i]->callback(cancellation_elements[i]);
      }
    }

    void BackgroundThread(std::shared_ptr<IteratorContext> ctx) {
      {
        mutex_lock l(mu_);
        background_thread_started_ = true;
      }
      int shard_to_fetch = 0;
      while (true) {
        HostBufferElement elem;
        bool end_of_iterator = false;

        {
          mutex_lock l(mu_);
          while (!cancellation_manager_.IsCancelled() &&
                 buffer_[shard_to_fetch].data.size() >= max_buffer_size_ &&
                 buffer_[shard_to_fetch].callbacks.empty()) {
            buffer_[shard_to_fetch].cond_var.wait(l);
          }

          if (cancellation_manager_.IsCancelled()) {
            background_thread_finished_ = true;
            shutdown_cond_var_.notify_all();
            return;
          }
        }

        elem.status = host_iterator_->GetNext(ctx.get(), &elem.value,
                                              &elem.end_of_sequence);

        if (elem.status.ok() && elem.end_of_sequence) {
          end_of_iterator = true;
        }

        std::shared_ptr<HostBuffer::CallbackContainer> callback_container;
        {
          mutex_lock l(mu_);
          // Try to find a callback, else just push stuff into buffer.
          if (!buffer_[shard_to_fetch].callbacks.empty()) {
            while (!buffer_[shard_to_fetch].callbacks.empty()) {
              if (buffer_[shard_to_fetch].callbacks.front()->is_called.exchange(
                      true)) {
                // This callback is already cancelled.
                buffer_[shard_to_fetch].callbacks.pop_front();
                continue;
              } else {
                callback_container =
                    std::move(buffer_[shard_to_fetch].callbacks.front());
                buffer_[shard_to_fetch].callbacks.pop_front();
                break;
              }
            }
          } else {
            buffer_[shard_to_fetch].data.push_back(std::move(elem));
            elem = HostBufferElement();
          }
        }

        if (callback_container) {
          if (callback_container->deregister_cancellation != nullptr) {
            callback_container->deregister_cancellation();
          }
          (*ctx->runner())(std::bind(std::move(callback_container->callback),
                                     std::move(elem)));
        }

        // Finish off the thread if we reach the end of the iterator. Runs
        // pending callbacks.
        if (end_of_iterator) {
          {
            mutex_lock l(mu_);
            background_thread_finished_ = true;
            end_of_iterator_ = true;
            shutdown_cond_var_.notify_all();
          }
          RunPendingCallbacks();
          return;
        }
        shard_to_fetch = (shard_to_fetch + 1) % size_;
      }
    }

    struct HostBuffer {
      condition_variable cond_var;
      std::deque<HostBufferElement> data;
      struct CallbackContainer {
        MultiDeviceIteratorCallback callback;
        // Whether callback is already called, either by the background thread
        // of by the cancellation callback.
        std::atomic<bool> is_called;
        std::function<void()> deregister_cancellation;
        explicit CallbackContainer(MultiDeviceIteratorCallback&& callback)
            : callback(std::move(callback)), is_called(false) {}
      };
      // The CallbackContainer is shared with the cancellation callback.
      std::deque<std::shared_ptr<CallbackContainer>> callbacks;
    };

    mutex mu_;
    bool background_thread_finished_ TF_GUARDED_BY(mu_) = false;
    bool background_thread_started_ TF_GUARDED_BY(mu_) = false;
    bool end_of_iterator_ TF_GUARDED_BY(mu_) = false;
    condition_variable shutdown_cond_var_ TF_GUARDED_BY(mu_);

    std::vector<HostBuffer> buffer_;

    const size_t size_;
    const int64_t max_buffer_size_;
    const int64_t incarnation_id_;
    CancellationManager cancellation_manager_;
    const std::unique_ptr<IteratorBase> host_iterator_;
    MultiDeviceIterator* const parent_;  // Not owned.
    std::unique_ptr<Thread> background_thread_ TF_GUARDED_BY(mu_);
  };

  IteratorMetricsCollector metrics_collector_;
  UnboundedThreadPool unbounded_thread_pool_;

  mutex mu_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::vector<string> devices_;
  const std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  FunctionLibraryRuntime* const flr_ = nullptr;  // not owned.
  const std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  const std::unique_ptr<FunctionHandleCache> function_handle_cache_;
  ResourceMgr resource_mgr_;
  CancellationManager cancellation_manager_;

  int64_t incarnation_id_ TF_GUARDED_BY(mu_) = 0;
  std::unique_ptr<MultiDeviceBuffer> multi_device_buffer_ TF_GUARDED_BY(mu_);
  core::RefCountPtr<DatasetBase> dataset_;
};

// Used to generate unique names for anonymous multi device iterators.
static std::atomic<int64_t> current_id_;

// Just creates a MultiDeviceIterator and returns it.
class MultiDeviceIteratorHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDevices, &devices_));
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel.
  ~MultiDeviceIteratorHandleOp() override {
    if (resource_ != nullptr) {
      resource_->Unref();
      if (cinfo_.resource_is_private_to_kernel()) {
        if (!cinfo_.resource_manager()
                 ->template Delete<MultiDeviceIterator>(cinfo_.container(),
                                                        cinfo_.name())
                 .ok()) {
          // Do nothing; the resource can have been deleted by session resets.
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    string unique_name = cinfo_.name();
    string container_name = cinfo_.container();
    {
      mutex_lock l(mu_);
      if (resource_ == nullptr) {
        FunctionLibraryRuntime* flr;
        std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
        OP_REQUIRES_OK(context, context->function_library()->Clone(
                                    &flib_def, &pflr, &flr));
        auto function_handle_cache = std::make_unique<FunctionHandleCache>(flr);
        ResourceMgr* mgr = context->resource_manager();
        OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

        MultiDeviceIterator* resource;

        if (name_ == ResourceHandle::ANONYMOUS_NAME) {
          unique_name = strings::StrCat("_AnonymousMultiDeviceIterator",
                                        current_id_.fetch_add(1));
          container_name = kAnonymousMultiDeviceIterator;
          resource = new MultiDeviceIterator(
              context->env(), output_types_, output_shapes_, devices_,
              std::move(flib_def), std::move(pflr), flr,
              std::move(function_handle_cache));
          // NOTE: `mgr->Create()` transfers the one reference on `resource` to
          // `mgr`.
          OP_REQUIRES_OK(context, mgr->Create<MultiDeviceIterator>(
                                      container_name, unique_name, resource));
        } else {
          unique_name = cinfo_.name();
          container_name = cinfo_.container();
          OP_REQUIRES_OK(context,
                         mgr->LookupOrCreate<MultiDeviceIterator>(
                             container_name, unique_name, &resource,
                             [this, context, flr, &flib_def, &pflr,
                              &function_handle_cache](MultiDeviceIterator** ret)
                                 TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                   *ret = new MultiDeviceIterator(
                                       context->env(), output_types_,
                                       output_shapes_, devices_,
                                       std::move(flib_def), std::move(pflr),
                                       flr, std::move(function_handle_cache));
                                   return absl::OkStatus();
                                 }));
          absl::Status s = VerifyResource(resource);
          if (TF_PREDICT_FALSE(!s.ok())) {
            resource->Unref();
            context->SetStatus(s);
            return;
          }
          resource_ = resource;
        }
      }
    }
    OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                context, 0, container_name, unique_name,
                                TypeIndex::Make<MultiDeviceIterator>()));
  }

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  absl::Status VerifyResource(MultiDeviceIterator* resource) {
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_types_, resource->output_types()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
    return absl::OkStatus();
  }

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  MultiDeviceIterator* resource_ TF_GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
  string name_;
  string container_;
  std::vector<string> devices_;
};

REGISTER_KERNEL_BUILDER(Name("MultiDeviceIterator").Device(DEVICE_CPU),
                        MultiDeviceIteratorHandleOp);

class AnonymousMultiDeviceIteratorOp
    : public AnonymousResourceOp<MultiDeviceIterator> {
 public:
  explicit AnonymousMultiDeviceIteratorOp(OpKernelConstruction* ctx)
      : AnonymousResourceOp<MultiDeviceIterator>(
            ctx,
            /* ref_counting */ true,
            /* Only V1 returns a deleter */
            /* return_deleter */
            ctx->def().op() == kAnonymousMultiDeviceIterator) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDevices, &devices_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

 private:
  string name() override { return kAnonymousMultiDeviceIterator; }

  absl::Status CreateResource(
      OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* lib, MultiDeviceIterator** resource) override {
    auto function_handle_cache = std::make_unique<FunctionHandleCache>(lib);
    *resource =
        new MultiDeviceIterator(ctx->env(), output_dtypes_, output_shapes_,
                                devices_, std::move(flib_def), std::move(pflr),
                                lib, std::move(function_handle_cache));
    return absl::OkStatus();
  }

  std::vector<string> devices_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name(kAnonymousMultiDeviceIterator).Device(DEVICE_CPU),
                        AnonymousMultiDeviceIteratorOp);
REGISTER_KERNEL_BUILDER(
    Name(kAnonymousMultiDeviceIteratorV3).Device(DEVICE_CPU),
    AnonymousMultiDeviceIteratorOp);

// Calls init on the MultiDeviceIterator.
class MultiDeviceIteratorInitOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorInitOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* tensor_max_buffer_size;
    OP_REQUIRES_OK(ctx, ctx->input("max_buffer_size", &tensor_max_buffer_size));
    int64_t max_buffer_size = tensor_max_buffer_size->scalar<int64_t>()();

    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
    core::RefCountPtr<MultiDeviceIterator> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &resource));

    IteratorContext::Params params(ctx);
    params.flr = resource->flr();
    params.function_handle_cache = resource->function_handle_cache();
    params.resource_mgr = resource->resource_mgr();
    params.cancellation_manager = resource->cancellation_manager();
    std::function<void()> deregister_fn;
    OP_REQUIRES_OK(
        ctx, RegisterCancellationCallback(
                 ctx->cancellation_manager(),
                 [cm = params.cancellation_manager]() { cm->StartCancel(); },
                 &deregister_fn));
    auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
    IteratorContext iter_ctx(std::move(params));

    std::unique_ptr<IteratorBase> iterator;
    DatasetBase* finalized_dataset;
    OP_REQUIRES_OK(ctx, FinalizeDataset(ctx, dataset, &finalized_dataset));
    OP_REQUIRES_OK(ctx, finalized_dataset->MakeIterator(std::move(iter_ctx),
                                                        /*parent=*/nullptr,
                                                        "Iterator", &iterator));
    core::ScopedUnref unref(finalized_dataset);
    int64_t incarnation_id;
    OP_REQUIRES_OK(ctx, resource->Init(std::move(iterator), max_buffer_size,
                                       &incarnation_id, dataset));
    Tensor tensor_incarnation_id(DT_INT64, TensorShape({}));
    tensor_incarnation_id.scalar<int64_t>()() = incarnation_id;
    OP_REQUIRES_OK(ctx,
                   ctx->set_output("incarnation_id", tensor_incarnation_id));
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiDeviceIteratorInit").Device(DEVICE_CPU),
                        MultiDeviceIteratorInitOp);

// Calls GetNextFromShard(shard) and returns a vector of Tensors as output.
class MultiDeviceIteratorGetNextFromShardOp : public AsyncOpKernel {
 public:
  explicit MultiDeviceIteratorGetNextFromShardOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(),
                           "tf_data_multi_device_iterator_get_next") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor* tensor_shard_num;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("shard_num", &tensor_shard_num), done);
    int32_t shard_num = tensor_shard_num->scalar<int32>()();

    const Tensor* tensor_incarnation_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->input("incarnation_id", &tensor_incarnation_id), done);
    int64_t incarnation_id = tensor_incarnation_id->scalar<int64_t>()();

    MultiDeviceIterator* iterator;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator), done);

    background_worker_.Schedule(std::bind(
        [ctx, iterator, shard_num, incarnation_id](DoneCallback done) {
          Notification n;
          absl::Time start_time = iterator->metrics_collector().RecordStart();
          MultiDeviceIteratorCallback callback = std::bind(
              [ctx, iterator, start_time, &n](const HostBufferElement& elem) {
                iterator->metrics_collector().RecordStop(start_time,
                                                         elem.value);
                absl::Status s = elem.status;
                if (!s.ok()) {
                  ctx->SetStatus(s);
                } else if (elem.end_of_sequence) {
                  ctx->SetStatus(errors::OutOfRange("End of sequence"));
                } else {
                  for (int i = 0; i < elem.value.size(); ++i) {
                    ctx->set_output(i, elem.value[i]);
                  }
                }
                n.Notify();
              },
              std::placeholders::_1);

          absl::Status s = iterator->GetNextFromShard(
              ctx, shard_num, incarnation_id, std::move(callback));
          if (!s.ok()) {
            ctx->SetStatus(s);
            iterator->Unref();
            done();
            return;
          }
          iterator->Unref();
          n.WaitForNotification();
          done();
        },
        std::move(done)));
  }

 private:
  BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorGetNextFromShard").Device(DEVICE_CPU),
    MultiDeviceIteratorGetNextFromShardOp);

class MultiDeviceIteratorToStringHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorToStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    // Validate that the handle corresponds to a real resource, and
    // that it is an MultiDeviceIterator.
    core::RefCountPtr<MultiDeviceIterator> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));

    Tensor* string_handle_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &string_handle_t));
    string_handle_t->scalar<tstring>()() =
        resource_handle_t.scalar<ResourceHandle>()().SerializeAsString();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorToStringHandle").Device(DEVICE_CPU),
    MultiDeviceIteratorToStringHandleOp);

class MultiDeviceIteratorFromStringHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorFromStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
    OP_REQUIRES(
        ctx,
        output_types_.empty() || output_shapes_.empty() ||
            output_types_.size() == output_shapes_.size(),
        errors::InvalidArgument("If both 'output_types' and 'output_shapes' "
                                "are set, they must have the same length."));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& string_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(string_handle_t.shape()),
                errors::InvalidArgument("string_handle must be a scalar"));

    ResourceHandle resource_handle;
    OP_REQUIRES(
        ctx,
        resource_handle.ParseFromString(string_handle_t.scalar<tstring>()()),
        errors::InvalidArgument(
            "Could not parse string_handle as a valid ResourceHandle"));

    OP_REQUIRES(
        ctx, resource_handle.device() == ctx->device()->attributes().name(),
        errors::InvalidArgument("Attempted create an iterator on device \"",
                                ctx->device()->attributes().name(),
                                "\" from handle defined on device \"",
                                resource_handle.device(), "\""));

    // Validate that the handle corresponds to a real resource, and
    // that it is an MultiDeviceIterator.
    core::RefCountPtr<MultiDeviceIterator> resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, resource_handle, &resource));
    if (!output_types_.empty()) {
      OP_REQUIRES_OK(ctx,
                     VerifyTypesMatch(output_types_, resource->output_types()));
    }
    if (!output_shapes_.empty()) {
      OP_REQUIRES_OK(ctx, VerifyShapesCompatible(output_shapes_,
                                                 resource->output_shapes()));
    }

    Tensor* resource_handle_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &resource_handle_t));
    resource_handle_t->scalar<ResourceHandle>()() = resource_handle;
  }

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorFromStringHandle").Device(DEVICE_CPU),
    MultiDeviceIteratorFromStringHandleOp);

class DeleteMultiDeviceIteratorOp : public OpKernel {
 public:
  explicit DeleteMultiDeviceIteratorOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    ResourceHandle handle = ctx->input(0).flat<ResourceHandle>()(0);
    // The iterator resource is guaranteed to
    // exist because the variant tensor wrapping the deleter is provided as an
    // unused input to this op, which guarantees that it has not run yet.
    OP_REQUIRES_OK(ctx, DeleteResource(ctx, handle));
  }
};

REGISTER_KERNEL_BUILDER(Name("DeleteMultiDeviceIterator").Device(DEVICE_CPU),
                        DeleteMultiDeviceIteratorOp);
// Since this op takes in Iterator handles as (unused) inputs, we don't want
// to constrain the iterator location to CPU only. Therefore, we exempt the
// colocation restriction for this op allowing the iterators to be placed on
// other devices.
REGISTER_INPUT_COLOCATION_EXEMPTION("DeleteMultiDeviceIterator");

}  // namespace
}  // namespace data
}  // namespace tensorflow
