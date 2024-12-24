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
#include "tensorflow/core/kernels/data/iterator_ops.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorflow/core/activity_watcher/activity.h"
#include "tensorflow/core/activity_watcher/activity_utils.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/finalization_utils.h"
#include "tensorflow/core/data/metric_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/tf_data_memory_logger.h"
#include "tensorflow/core/data/tfdataz_metrics.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/optional_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following ops.

const char kAnonymousIterator[] = "AnonymousIterator";
const char kAnonymousIteratorV2[] = "AnonymousIteratorV2";
const char kAnonymousIteratorV3[] = "AnonymousIteratorV3";
const char kIteratorVariantTypeName[] = "tensorflow::Iterator";
const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";

bool SymbolicCheckpointEnabled(const Options& options) {
  return options.optional_symbolic_checkpoint_case() ==
             Options::kSymbolicCheckpoint &&
         options.symbolic_checkpoint();
}

}  // namespace

/* static */ constexpr const char* const
    SerializeIteratorOp::kExternalStatePolicy;

IteratorResource::IteratorResource(
    Env* env, const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes,
    std::unique_ptr<DeviceMgr> device_mgr,
    std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* flr)
    : metrics_collector_(flr->device()->device_type(), *env),
      unbounded_thread_pool_(env, "tf_data_iterator_resource"),
      env_(*env),
      device_mgr_(std::move(device_mgr)),
      iterator_state_(std::make_shared<State>(std::move(flib_def),
                                              std::move(pflr), flr,
                                              /*iterator=*/nullptr)),
      output_dtypes_(output_dtypes),
      output_shapes_(output_shapes) {
  VLOG(2) << "creating iterator resource";
}

IteratorResource::~IteratorResource() {
  TfDatazMetricsRegistry::Deregister(tf_dataz_metrics_collector_);
  VLOG(2) << "destroying iterator resource";
}

absl::Status IteratorResource::GetNext(OpKernelContext* ctx,
                                       std::vector<Tensor>* out_tensors,
                                       bool* end_of_sequence) {
  std::shared_ptr<State> captured_state;
  {
    tf_shared_lock l(mu_);
    captured_state = iterator_state_;
  }
  auto iterator = captured_state->iterator();
  if (!iterator) {
    return errors::FailedPrecondition(
        "GetNext() failed because the iterator has not been initialized. "
        "Ensure that you have run the initializer operation for this iterator "
        "before getting the next element.");
  }
  auto* dataset = captured_state->dataset();
  IteratorContext::Params params(ctx);
  params.cancellation_manager = captured_state->cancellation_manager();
  params.flr = captured_state->flr();
  params.function_handle_cache = captured_state->function_handle_cache();
  params.resource_mgr = captured_state->resource_mgr();
  params.symbolic_checkpoint = SymbolicCheckpointEnabled(dataset->options());
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.id_registry = captured_state->id_registry();
  params.warm_start = dataset->options().warm_start();
  params.model = captured_state->model();
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
  IteratorContext iter_ctx(std::move(params));
  const absl::Time start_time = metrics_collector_.RecordStart();
  auto status = iterator->GetNext(&iter_ctx, out_tensors, end_of_sequence);
  metrics_collector_.RecordStop(start_time, *out_tensors);
  const int64_t get_next_latency_micros =
      env_.NowMicros() - absl::ToUnixMicros(start_time);
  tf_dataz_metrics_collector_->RecordGetNextLatency(get_next_latency_micros);
  captured_state->MergeCheckpoint(iter_ctx.checkpoint());
  return status;
}

absl::Status IteratorResource::GetModelProto(std::string& model_proto) {
  std::shared_ptr<State> captured_state;
  {
    tf_shared_lock l(mu_);
    captured_state = iterator_state_;
  }
  auto iterator = captured_state->iterator();
  if (!iterator) {
    return absl::FailedPreconditionError(
        "GetModelProto() failed because the iterator has not been initialized. "
        "Ensure that you have run the initializer operation for this iterator "
        "before getting the next element.");
  }

  model::ModelProto proto;
  if (auto model = captured_state->model(); model) {
    TF_RETURN_IF_ERROR(model->ToProto(&proto));
  } else {
    return absl::NotFoundError(
        "Cannot find this iterator's analytical model. Did you disable "
        "autotune for the dataset used to create this iterator? See more "
        "information at "
        "https://www.tensorflow.org/api_docs/python/tf/data/experimental/"
        "AutotuneOptions .");
  }
  model_proto = proto.SerializeAsString();
  return absl::OkStatus();
}

absl::Status IteratorResource::Save(OpKernelContext* ctx,
                                    ExternalStatePolicy external_state_policy,
                                    IteratorStateWriter* writer) {
  std::shared_ptr<State> captured_state;
  {
    tf_shared_lock l(mu_);
    captured_state = iterator_state_;
  }
  auto iterator = captured_state->iterator();
  if (!iterator) {
    return errors::FailedPrecondition(
        "Save() failed because the iterator has not been initialized. Ensure "
        "that you have run the initializer operation for this iterator before "
        "saving it.");
  }
  auto* dataset = captured_state->dataset();
  if (SymbolicCheckpointEnabled(dataset->options())) {
    const auto& checkpoint = captured_state->checkpoint();
    if (!checkpoint.GetStatus().ok()) {
      LOG(WARNING) << "Symbolic checkpointing failed: "
                   << checkpoint.GetStatus();
      return checkpoint.GetStatus();
    }
    LOG(INFO) << "Saving symbolic checkpoint";
    TF_RETURN_IF_ERROR(checkpoint.Save(writer));
    return absl::OkStatus();
  }
  SerializationContext::Params params(ctx);
  params.external_state_policy = external_state_policy;
  params.symbolic_checkpoint = SymbolicCheckpointEnabled(dataset->options());
  SerializationContext serialization_ctx(params);
  return iterator->Save(&serialization_ctx, writer);
}

absl::Status IteratorResource::Restore(OpKernelContext* ctx,
                                       IteratorStateReader* reader) {
  const DatasetBase* dataset;
  std::shared_ptr<State> new_state;
  const DatasetBase* input_dataset;
  {
    tf_shared_lock l(mu_);
    auto iterator = iterator_state_->iterator();
    if (!iterator) {
      return errors::FailedPrecondition(
          "Restore() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before restoring it.");
    }
    dataset = iterator->dataset();
    // Hang onto a reference until we've created the new iterator, which will
    // then hold its own reference to keep the dataset alive.
    dataset->Ref();
    new_state =
        std::make_shared<State>(iterator_state_->flib_def(),
                                iterator_state_->pflr(), iterator_state_->flr(),
                                /*iterator=*/nullptr);
    input_dataset = iterator_state_->dataset();

    // This is to ensure the checkpoint can be restored correctly
    // without worrying thread interleaving events.
    // For example, `GlobalShuffleDatasetOp::Dataset::Iterator::Initialize`
    // could be stateful due to the seed generator.
    // Therefore, before restoring from the checkpoint, we need to make
    // sure cancellation is marked so that
    // `GlobalShuffleDatasetOp::Dataset::Iterator::Initialize` would know not to
    // execute anymore stateful operations like seed generation.
    iterator_state_->cancellation_manager()->StartCancel();
  }
  core::ScopedUnref scoped_unref(dataset);
  IteratorContext::Params params(ctx);
  params.cancellation_manager = new_state->cancellation_manager();
  params.flr = new_state->flr();
  params.function_handle_cache = new_state->function_handle_cache();
  params.resource_mgr = new_state->resource_mgr();
  params.symbolic_checkpoint =
      SymbolicCheckpointEnabled(input_dataset->options());
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.id_registry = new_state->id_registry();
  params.warm_start = dataset->options().warm_start();
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
  IteratorContext iter_ctx(IteratorContext(std::move(params)));
  std::unique_ptr<IteratorBase> iterator_base;
  TF_RETURN_IF_ERROR(dataset->MakeIteratorFromCheckpoint(
      &iter_ctx, "Iterator", reader, &iterator_base));
  new_state->DowncastAndSetIteratorAndDataset(std::move(iterator_base),
                                              input_dataset);
  new_state->MergeCheckpoint(iter_ctx.checkpoint());
  mutex_lock l(mu_);
  std::swap(iterator_state_, new_state);
  return absl::OkStatus();
}

absl::Status IteratorResource::SetIteratorFromDataset(
    OpKernelContext* ctx, const DatasetBase* dataset) {
  std::shared_ptr<State> new_state;
  {
    tf_shared_lock l(mu_);
    new_state =
        std::make_shared<State>(iterator_state_->flib_def(),
                                iterator_state_->pflr(), iterator_state_->flr(),
                                /*iterator=*/nullptr);
  }

  // Create new iterator.
  IteratorContext::Params params(ctx);
  params.cancellation_manager = new_state->cancellation_manager();
  params.flr = new_state->flr();
  params.function_handle_cache = new_state->function_handle_cache();
  params.resource_mgr = new_state->resource_mgr();
  params.symbolic_checkpoint = SymbolicCheckpointEnabled(dataset->options());
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.id_registry = new_state->id_registry();
  params.warm_start = dataset->options().warm_start();
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
  IteratorContext iter_ctx(IteratorContext(std::move(params)));
  std::unique_ptr<IteratorBase> iterator;
  if (ctx->function_library()->device()->device_type() == DEVICE_CPU) {
    DatasetBase* finalized_dataset;
    TF_ASSIGN_OR_RETURN(finalized_dataset, GetFinalizedDataset(ctx, dataset));
    TF_RETURN_IF_ERROR(finalized_dataset->MakeIterator(&iter_ctx,
                                                       /*parent=*/nullptr,
                                                       "Iterator", &iterator));
  } else {
    TF_RETURN_IF_ERROR(dataset->MakeIterator(&iter_ctx,
                                             /*parent=*/nullptr, "Iterator",
                                             &iterator));
  }
  TF_RETURN_IF_ERROR(
      VerifyTypesMatch(output_dtypes_, iterator->output_dtypes()));
  TF_RETURN_IF_ERROR(
      VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));
  new_state->DowncastAndSetIteratorAndDataset(std::move(iterator), dataset);
  new_state->SetModel(iter_ctx.model());
  new_state->MergeCheckpoint(iter_ctx.checkpoint());
  mutex_lock l(mu_);
  std::swap(iterator_state_, new_state);
  tf_dataz_metrics_collector_ = std::make_shared<TfDatazMetricsCollector>(
      env_, iterator_state_->iterator(), iterator_state_->model());
  EnsureIteratorMemoryLoggerStarted();
  TfDatazMetricsRegistry::Register(tf_dataz_metrics_collector_);
  return absl::OkStatus();
}

void IteratorResource::State::DowncastAndSetIteratorAndDataset(
    std::unique_ptr<IteratorBase> it, const DatasetBase* dataset) {
  iterator_.reset(static_cast<DatasetBaseIterator*>(it.release()));
  if (dataset) {
    dataset->Ref();
    dataset_.reset(const_cast<DatasetBase*>(dataset));
  }
}

void IteratorResource::State::MergeCheckpoint(MemoryCheckpoint* other) {
  if (SymbolicCheckpointEnabled(dataset_->options())) {
    checkpoint_.Merge(other);
  }
}

void IteratorResource::State::SetModel(std::shared_ptr<model::Model> model) {
  model_ = model;
}

namespace {

// A helper class that uses a list of IteratorStateVariant objects to represent
// the state for an iterator resource. It exposes methods that help with
// saving and restoring of this state. Sample usage
// Saving:
//   IteratorVariantSerializer serializer;
//   serializer.InitializeFromIterator(iterator_resource);
//   Tensor serialized_t;
//   serializer.Serialize(&serialized_t);
//
// Restoring:
//   IteratorVariantSerializer serializer;
//   serializer.InitFromTensor(ctx->input(0));
//   IteratorStateReader* reader = serializer.GetReader();
//   iterator_resource->Restore(ctx, reader);
class IteratorVariantSerializer {
 public:
  IteratorVariantSerializer() = default;

  // Calls `Save` on the iterator_resource to build up the list of
  // IteratorStateVariant objects.
  absl::Status InitializeFromIterator(OpKernelContext* ctx,
                                      ExternalStatePolicy external_state_policy,
                                      IteratorResource* iterator_resource) {
    VariantTensorDataWriter writer;
    TF_RETURN_IF_ERROR(
        iterator_resource->Save(ctx, external_state_policy, &writer));
    std::vector<std::unique_ptr<VariantTensorData>> data;
    writer.ReleaseData(&data);
    variants_.clear();
    variants_.reserve(data.size());
    for (auto& it : data) {
      IteratorStateVariant v;
      TF_RETURN_IF_ERROR(v.InitializeFromVariantData(std::move(it)));
      variants_.push_back(v);
    }
    num_tensors_ = variants_.size();
    can_serialize_ = true;
    return absl::OkStatus();
  }

  // Initializes `this` from `serialized_t` while restoring the iterator state.
  absl::Status InitFromTensor(const Tensor* serialized_t) {
    int64_t num_tensors = serialized_t->dim_size(0);
    auto serialized_vec = serialized_t->vec<Variant>();
    std::vector<const VariantTensorData*> data;
    data.reserve(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
      auto* w = serialized_vec(i).get<IteratorStateVariant>();
      if (!w) {
        return errors::Internal(
            "Cannot initialize an iterator from tensor ",
            serialized_vec(i).DebugString(),
            ". Expected a variant tensor of type IteratorStateVariant");
      }
      data.push_back(w->GetData());
    }
    reader_ = std::make_unique<VariantTensorDataReader>(data);
    num_tensors_ = data.size();
    return absl::OkStatus();
  }

  int64_t NumTensors() { return num_tensors_; }

  // Stores the IteratorStateVariant list into a pre-allocated tensor. Expects
  // that InitializeFromIterator was called before.
  absl::Status Serialize(Tensor* serialized) {
    if (!can_serialize_) {
      return errors::InvalidArgument(
          "Please call InitializeFromIterator before calling Serialize.");
    }
    int64_t size = variants_.size();
    for (int64_t i = 0; i < size; ++i) {
      if (variants_[i].GetData() == nullptr) {
        return errors::Internal(
            "Cannot serialize an empty IteratorStateVariant");
      }
      serialized->vec<Variant>()(i) = variants_[i];
    }
    return absl::OkStatus();
  }

  // Returns an IteratorStateReader to restore iterator state. Expects that
  // InitFromTensor was called before.
  IteratorStateReader* GetReader() { return reader_.get(); }

 private:
  bool can_serialize_ = false;
  int64_t num_tensors_;
  std::vector<IteratorStateVariant> variants_;
  std::unique_ptr<IteratorStateReader> reader_;
};

}  // namespace

// Note that IteratorHandleOp holds a reference to the resource it creates. If
// cleaning up resources with DestroyResourceOp is important, consider creating
// resource containers with AnonymousIteratorHandleOp instead.
IteratorHandleOp::IteratorHandleOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
}

// The resource is deleted from the resource manager only when it is private
// to kernel. Ideally the resource should be deleted when it is no longer held
// by anyone, but it would break backward compatibility.
IteratorHandleOp::~IteratorHandleOp() {
  if (resource_ != nullptr) {
    resource_->Unref();
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->template Delete<IteratorResource>(cinfo_.container(),
                                                   cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }
}

void IteratorHandleOp::Compute(OpKernelContext* context)
    TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(mu_);
    if (resource_ == nullptr) {
      FunctionLibraryRuntime* flr;
      std::unique_ptr<DeviceMgr> device_mgr(nullptr);
      std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
      // If the iterator is shared then we construct a new FLR, and pass that
      // in. NOTE(mrry,rohanj): In this case it is not possible to call remote
      // functions from the iterator. We may add this functionality if there
      // is sufficient demand, but it will require a significant refactoring.
      if (!name_.empty()) {
        flr = CreatePrivateFLR(context, &device_mgr, &flib_def, &pflr);
      } else {
        OP_REQUIRES_OK(context, context->function_library()->Clone(
                                    &flib_def, &pflr, &flr, true));
      }

      ResourceMgr* mgr = context->resource_manager();
      OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

      IteratorResource* resource;
      OP_REQUIRES_OK(
          context,
          mgr->LookupOrCreate<IteratorResource>(
              cinfo_.container(), cinfo_.name(), &resource,
              [context, flr, &device_mgr, &flib_def, &pflr,
               this](IteratorResource** ret) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                *ret = new IteratorResource(
                    context->env(), output_dtypes_, output_shapes_,
                    std::move(device_mgr), std::move(flib_def), std::move(pflr),
                    flr);
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
  OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                              context, 0, cinfo_.container(), cinfo_.name(),
                              TypeIndex::Make<IteratorResource>()));
}

absl::Status IteratorHandleOp::VerifyResource(IteratorResource* resource) {
  TF_RETURN_IF_ERROR(
      VerifyTypesMatch(output_dtypes_, resource->output_dtypes()));
  TF_RETURN_IF_ERROR(
      VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
  return absl::OkStatus();
}

FunctionLibraryRuntime* IteratorHandleOp::CreatePrivateFLR(
    OpKernelContext* ctx, std::unique_ptr<DeviceMgr>* device_mgr,
    std::unique_ptr<FunctionLibraryDefinition>* flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime>* pflr) {
  // Wrap the existing device in order to see any captured resources
  // in its resource manager. The existing device will outlive the
  // IteratorResource, because we are storing the IteratorResource
  // in that device's resource manager.

  *device_mgr =
      std::make_unique<StaticDeviceMgr>(RenamedDevice::NewRenamedDevice(
          ctx->device()->name(), down_cast<Device*>(ctx->device()),
          false /* owns_underlying */, false /* isolate_session_state */));
  *flib_def = std::make_unique<FunctionLibraryDefinition>(
      *ctx->function_library()->GetFunctionLibraryDefinition());
  const auto* config = ctx->function_library()->config_proto();
  *pflr = std::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr->get(), ctx->env(),
      /*config=*/config, graph_def_version_, flib_def->get(),
      config->graph_options().optimizer_options());

  return (*pflr)->GetFLR(ctx->device()->name());
}

// Like IteratorHandleOp, but creates handles which are never shared, and does
// not hold a reference to these handles. The latter is important for eager
// execution, since OpKernel instances generally live as long as the program
// running them.
AnonymousIteratorHandleOp::AnonymousIteratorHandleOp(
    OpKernelConstruction* context)
    : AnonymousResourceOp<IteratorResource>(
          context,
          /* ref_counting */
          // Only enable this for V2 (via Python's iter protocol),
          // AnonymousIteratorV1 requires IteratorToStringHandle, which is
          // undefined on Refcounting ResourceHandle.
          context->def().op() == kAnonymousIteratorV2 ||
              context->def().op() == kAnonymousIteratorV3,
          // V1 does not return a deleter.
          /* return_deleter */
          context->def().op() == kAnonymousIteratorV2),
      graph_def_version_(context->graph_def_version()) {
  OP_REQUIRES_OK(context, context->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(context, context->GetAttr(kOutputShapes, &output_shapes_));
}

string AnonymousIteratorHandleOp::name() { return kAnonymousIterator; }

absl::Status AnonymousIteratorHandleOp::CreateResource(
    OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, IteratorResource** resource) {
  std::unique_ptr<DeviceMgr> device_mgr(nullptr);
  *resource = new IteratorResource(ctx->env(), output_dtypes_, output_shapes_,
                                   std::move(device_mgr), std::move(flib_def),
                                   std::move(pflr), lib);
  return absl::OkStatus();
}

HybridAsyncOpKernel::HybridAsyncOpKernel(OpKernelConstruction* ctx,
                                         const char* background_worker_name)
    : AsyncOpKernel(ctx),
      background_worker_(ctx->env(), background_worker_name) {}

void HybridAsyncOpKernel::ComputeAsync(OpKernelContext* ctx,
                                       DoneCallback done) {
  background_worker_.Schedule([this, ctx, done = std::move(done)]() {
    ctx->SetStatus(DoCompute(ctx));
    done();
  });
}

void HybridAsyncOpKernel::Compute(OpKernelContext* ctx) {
  ctx->SetStatus(DoCompute(ctx));
}

absl::Status MakeIteratorOp::DoCompute(OpKernelContext* ctx) {
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  IteratorResource* iterator_resource;
  TF_RETURN_IF_ERROR(
      LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  return iterator_resource->SetIteratorFromDataset(ctx, dataset);
}

absl::Status DeleteIteratorOp::DoCompute(OpKernelContext* ctx) {
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  const ResourceHandle& handle = ctx->input(0).flat<ResourceHandle>()(0);
  // The iterator resource is guaranteed to exist because the variant tensor
  // wrapping the deleter is provided as an unused input to this op, which
  // guarantees that it has not run yet.
  return DeleteResource(ctx, handle);
}

namespace {

class ToSingleElementOp : public AsyncOpKernel {
 public:
  explicit ToSingleElementOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        metrics_collector_(ctx->device()->attributes().device_type(),
                           *ctx->env()),
        unbounded_threadpool_(ctx->env(), "tf_data_to_single_element") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    unbounded_threadpool_.Schedule([this, ctx, done = std::move(done)]() {
      ctx->SetStatus(DoCompute(ctx));
      done();
    });
  }

  void Compute(OpKernelContext* ctx) override {
    ctx->SetStatus(DoCompute(ctx));
  }

 private:
  absl::Status DoCompute(OpKernelContext* ctx) {
    tsl::profiler::TraceMe traceme(
        [&] {
          return tsl::profiler::TraceMeEncode("ToSingleElementOp::DoCompute",
                                              {{"id", ctx->step_id()}});
        },
        profiler::kInfo);
    tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                   ctx->op_kernel().type_string());
    metrics::RecordTFDataFetchOp("ToSingleElementOp");
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    IteratorContext::Params params(ctx);
    ResourceMgr resource_mgr;
    params.resource_mgr = &resource_mgr;
    CancellationManager cancellation_manager(ctx->cancellation_manager());
    params.cancellation_manager = &cancellation_manager;

    IteratorContext iter_ctx(std::move(params));
    std::unique_ptr<IteratorBase> iterator;
    TF_RETURN_IF_ERROR(dataset->MakeIterator(
        &iter_ctx, /*parent=*/nullptr, "SingleElementIterator", &iterator));

    std::vector<Tensor> components;
    components.reserve(dataset->output_dtypes().size());
    bool end_of_sequence = false;

    const absl::Time start_time = metrics_collector_.RecordStart();
    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &components, &end_of_sequence));
    metrics_collector_.RecordStop(start_time, components);

    if (end_of_sequence) {
      return errors::InvalidArgument("Dataset was empty.");
    }
    TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, components));
    TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, components));
    for (int i = 0; i < components.size(); ++i) {
      ctx->set_output(i, components[i]);
    }

    components.clear();
    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &components, &end_of_sequence));
    if (!end_of_sequence) {
      return errors::InvalidArgument("Dataset had more than one element.");
    }
    return absl::OkStatus();
  }

  IteratorMetricsCollector metrics_collector_;
  UnboundedThreadPool unbounded_threadpool_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class OneShotIteratorOp : public AsyncOpKernel {
 public:
  explicit OneShotIteratorOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_one_shot_iterator"),
        graph_def_version_(ctx->graph_def_version())

  {
    string shared_name;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name));
    OP_REQUIRES(ctx, shared_name.empty(),
                errors::InvalidArgument("OneShotIteratorOp does not currently "
                                        "support the 'shared_name' attr."));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("dataset_factory", &dataset_factory_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

  ~OneShotIteratorOp() override {
    if (iterator_resource_ != nullptr) {
      iterator_resource_->Unref();
      if (!cinfo_.resource_manager()
               ->Delete<IteratorResource>(cinfo_.container(), cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  // NOTE(mrry): This is based on `ResourceOpKernel<T>::Compute()`,
  // but due to the fact that `ResourceOpKernel<T>::CreateResource()`
  // does not provide access to the `OpKernelContext*` and we need
  // this to invoke the factory function, it's not possible to
  // implement this kernel by implementing `CreateResource()`.
  // Furthermore, due to the fact that this kernel might block when
  // running the initialization function, we must implement this
  // kernel as an async kernel.
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                   ctx->op_kernel().type_string());
    {
      mutex_lock l(mu_);
      if (iterator_resource_ == nullptr && initialization_status_.ok()) {
        // The initialization thread will call `done`.
        if (!initialization_started_) {
          // TODO(mrry): Convert the initialization code to use
          // callbacks instead of wasting a thread.
          background_worker_.Schedule([this, ctx, done]() { Init(ctx, done); });
          initialization_started_ = true;
        } else {
          done_callbacks_.emplace_back(ctx, std::move(done));
        }
        return;
      }
    }
    ProduceOutput(ctx, done);
  }

 private:
  void Init(OpKernelContext* ctx, const DoneCallback& done) {
    IteratorResource* iterator = nullptr;
    ContainerInfo cinfo;
    absl::Status s = TryInit(ctx, &iterator, &cinfo);

    std::vector<std::pair<OpKernelContext*, DoneCallback>> callbacks_to_run;
    {
      mutex_lock l(mu_);
      if (s.ok()) {
        iterator_resource_ = iterator;
        cinfo_ = cinfo;
      }
      initialization_status_ = s;
      std::swap(done_callbacks_, callbacks_to_run);
    }

    for (auto&& ctx_done : callbacks_to_run) {
      ProduceOutput(ctx_done.first, ctx_done.second);
    }
    ProduceOutput(ctx, done);
  }

  absl::Status TryInit(OpKernelContext* ctx, IteratorResource** iterator,
                       ContainerInfo* cinfo) {
    TF_RETURN_IF_ERROR(cinfo->Init(ctx->resource_manager(), def()));

    FunctionLibraryRuntime* flr;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    TF_RETURN_IF_ERROR(
        ctx->function_library()->Clone(&flib_def, &pflr, &flr, true));

    // Create an IteratorResource that will hold the iterator for this op.
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()->LookupOrCreate<IteratorResource>(
            cinfo->container(), cinfo->name(), iterator,
            [ctx, flr, this, &flib_def, &pflr](IteratorResource** ret)
                TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                  *ret = new IteratorResource(
                      ctx->env(), output_dtypes_, output_shapes_,
                      /*device_mgr=*/nullptr, std::move(flib_def),
                      std::move(pflr), flr);
                  return absl::OkStatus();
                }));

    core::ScopedUnref unref_iterator(*iterator);

    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, (*iterator)->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, (*iterator)->output_shapes()));

    // Call the dataset_factory_func_ to create a new dataset,
    // over which this op will iterate.
    FunctionLibraryRuntime::Handle f_handle;
    TF_RETURN_IF_ERROR(ctx->function_library()->Instantiate(
        dataset_factory_func_.name(), AttrSlice(&dataset_factory_func_.attr()),
        &f_handle));
    FunctionLibraryRuntime::Options opts;
    opts.cancellation_manager = ctx->cancellation_manager();
    ScopedStepContainer step_container(opts.step_id, [ctx](const string& name) {
      ctx->resource_manager()->Cleanup(name).IgnoreError();
    });
    opts.step_container = &step_container;
    opts.runner = ctx->runner();
    opts.run_all_kernels_inline = ctx->run_all_kernels_inline();
    std::vector<Tensor> return_values;
    TF_RETURN_IF_ERROR(ctx->function_library()->RunSync(
        std::move(opts), f_handle, {}, &return_values));
    if (return_values.size() != 1 || return_values[0].dtype() != DT_VARIANT ||
        !TensorShapeUtils::IsScalar(return_values[0].shape())) {
      return errors::InvalidArgument(
          "The `dataset_factory` function must return "
          "a single scalar of dtype DT_VARIANT.");
    }

    // Create an iterator for the dataset that was created in the
    // factory function.
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(return_values[0], &dataset));
    TF_RETURN_IF_ERROR((*iterator)->SetIteratorFromDataset(ctx, dataset));
    (*iterator)->Ref();
    return absl::OkStatus();
  }

  void ProduceOutput(OpKernelContext* ctx, const DoneCallback& done) {
    Tensor* handle;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, TensorShape({}), &handle),
                         done);
    absl::Status s;
    {
      mutex_lock l(mu_);
      s = initialization_status_;
      if (s.ok()) {
        handle->scalar<ResourceHandle>()() =
            MakeResourceHandle<IteratorResource>(ctx, cinfo_.container(),
                                                 cinfo_.name());
      }
    }
    OP_REQUIRES_OK_ASYNC(ctx, s, done);
    done();
  }

  NameAttrList dataset_factory_func_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;

  BackgroundWorker background_worker_;

  mutex mu_;
  ContainerInfo cinfo_ TF_GUARDED_BY(mu_);
  IteratorResource* iterator_resource_ TF_GUARDED_BY(mu_) = nullptr;

  bool initialization_started_ TF_GUARDED_BY(mu_) = false;
  absl::Status initialization_status_ TF_GUARDED_BY(mu_);
  std::vector<std::pair<OpKernelContext*, DoneCallback>> done_callbacks_
      TF_GUARDED_BY(mu_);
  const int graph_def_version_;
};

}  // namespace

AsyncOpKernel* IteratorGetNextOp::AsAsync() {
  return type_string() == "IteratorGetNextSync" ? nullptr : this;
}

void RecordElementSize(const std::vector<Tensor> element,
                       tsl::profiler::TraceMe* traceme) {
  traceme->AppendMetadata([&]() {
    int64_t element_size = 0;
    for (const auto& component : element) {
      element_size += component.TotalBytes();
    }
    return tsl::profiler::TraceMeEncode({{"element_size", element_size}});
  });
}

absl::Status IteratorGetNextOp::DoCompute(OpKernelContext* ctx) {
  VLOG(3) << "IteratorGetNextOp enter. iter_id=" << ctx->frame_iter().iter_id;
  auto cleanup = gtl::MakeCleanup([ctx] {
    VLOG(3) << "IteratorGetNextOp exit. iter_id=" << ctx->frame_iter().iter_id;
  });
  activity_watcher::ActivityScope activity_scope([ctx = ctx]() {
    return activity_watcher::ActivityFromContext(
        ctx, "IteratorGetNextOp::DoCompute",
        activity_watcher::ActivityCategory::kDatasetOp);
  });
  tsl::profiler::TraceMe traceme(
      [&] {
        return tsl::profiler::TraceMeEncode(
            "IteratorGetNextOp::DoCompute",
            {{"id", ctx->step_id()}, {"iter_num", ctx->frame_iter().iter_id}});
      },
      profiler::kInfo);
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  metrics::RecordTFDataFetchOp("IteratorGetNextOp");
  IteratorResource* iterator;
  TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
  core::ScopedUnref unref_iterator(iterator);
  std::vector<Tensor> components;
  bool end_of_sequence = false;

  TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &components, &end_of_sequence));
  if (end_of_sequence) {
    return errors::OutOfRange("End of sequence");
  }
  TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, components));
  TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, components));
  RecordElementSize(components, &traceme);
  for (int i = 0; i < components.size(); ++i) {
    ctx->set_output(i, components[i]);
  }
  return absl::OkStatus();
}

absl::Status IteratorGetModelProtoOp::DoCompute(OpKernelContext* ctx) {
  IteratorResource* iterator = nullptr;
  TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
  core::ScopedUnref unref_iterator(iterator);

  std::string model_proto;
  TF_RETURN_IF_ERROR(iterator->GetModelProto(model_proto));
  Tensor* model_proto_result;
  TF_RETURN_IF_ERROR(
      ctx->allocate_output(0, TensorShape({}), &model_proto_result));
  model_proto_result->scalar<tstring>()() = model_proto;
  return absl::OkStatus();
}

absl::Status IteratorGetNextAsOptionalOp::DoCompute(OpKernelContext* ctx) {
  VLOG(3) << "IteratorGetNextAsOptionalOp enter. iter_id="
          << ctx->frame_iter().iter_id;
  auto cleanup = gtl::MakeCleanup([ctx] {
    VLOG(3) << "IteratorGetNextAsOptionalOp exit. iter_id="
            << ctx->frame_iter().iter_id;
  });
  activity_watcher::ActivityScope activity_scope([ctx = ctx]() {
    return activity_watcher::ActivityFromContext(
        ctx, "IteratorGetNextAsOptionalOp::DoCompute",
        activity_watcher::ActivityCategory::kDatasetOp);
  });
  tsl::profiler::TraceMe traceme(
      [&] {
        return tsl::profiler::TraceMeEncode(
            "IteratorGetNextAsOptionalOp::DoCompute",
            {{"id", ctx->step_id()}, {"iter_num", ctx->frame_iter().iter_id}});
      },
      profiler::kInfo);
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  metrics::RecordTFDataFetchOp("IteratorGetNextAsOptionalOp");
  IteratorResource* iterator;
  TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
  core::ScopedUnref unref_iterator(iterator);
  std::vector<Tensor> components;
  bool end_of_sequence = false;

  TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &components, &end_of_sequence));

  if (end_of_sequence) {
    return WriteOptionalNoneToOutput(ctx, 0);
  } else {
    RecordElementSize(components, &traceme);
    for (int i = 0; i < components.size(); ++i) {
      if (components[i].dtype() != output_types_[i]) {
        return errors::InvalidArgument(
            "The given optional does not match the expected type for "
            "component ",
            i, ". Expected: ", DataTypeString(output_types_[i]),
            ". Actual: ", DataTypeString(components[i].dtype()), ".");
      }
      if (!output_shapes_[i].IsCompatibleWith(components[i].shape())) {
        return errors::InvalidArgument(
            "The given optional does not match the expected shape "
            "for component ",
            i, ". Expected: ", output_shapes_[i].DebugString(),
            ". Actual: ", components[i].shape().DebugString(), ".");
      }
    }
    return WriteOptionalWithValueToOutput(ctx, 0, std::move(components));
  }
}

void IteratorToStringHandleOp::Compute(OpKernelContext* ctx) {
  const Tensor& resource_handle_t = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
              errors::InvalidArgument("resource_handle must be a scalar"));

  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator_resource));
  iterator_resource->Unref();

  Tensor* string_handle_t;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({}), &string_handle_t));
  string_handle_t->scalar<tstring>()() =
      resource_handle_t.scalar<ResourceHandle>()().SerializeAsString();
}

IteratorFromStringHandleOp::IteratorFromStringHandleOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES(
      ctx,
      output_dtypes_.empty() || output_shapes_.empty() ||
          output_dtypes_.size() == output_shapes_.size(),
      errors::InvalidArgument("If both 'output_types' and 'output_shapes' "
                              "are set, they must have the same length."));
}

void IteratorFromStringHandleOp::Compute(OpKernelContext* ctx) {
  const Tensor& string_handle_t = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(string_handle_t.shape()),
              errors::InvalidArgument("string_handle must be a scalar"));

  ResourceHandle resource_handle;
  OP_REQUIRES(
      ctx, resource_handle.ParseFromString(string_handle_t.scalar<tstring>()()),
      errors::InvalidArgument(
          "Could not parse string_handle as a valid ResourceHandle"));

  OP_REQUIRES(
      ctx, resource_handle.device() == ctx->device()->attributes().name(),
      errors::InvalidArgument("Attempted create an iterator on device \"",
                              ctx->device()->attributes().name(),
                              "\" from handle defined on device \"",
                              resource_handle.device(), "\""));

  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, resource_handle, &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  if (!output_dtypes_.empty()) {
    OP_REQUIRES_OK(ctx, VerifyTypesMatch(output_dtypes_,
                                         iterator_resource->output_dtypes()));
  }
  if (!output_shapes_.empty()) {
    OP_REQUIRES_OK(ctx,
                   VerifyShapesCompatible(output_shapes_,
                                          iterator_resource->output_shapes()));
  }

  Tensor* resource_handle_t;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({}), &resource_handle_t));
  resource_handle_t->scalar<ResourceHandle>()() = resource_handle;
}

SerializeIteratorOp::SerializeIteratorOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  if (ctx->HasAttr(kExternalStatePolicy)) {
    int64_t external_state_policy;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr(kExternalStatePolicy, &external_state_policy));
    external_state_policy_ = ExternalStatePolicy(external_state_policy);
  }
}

void SerializeIteratorOp::Compute(OpKernelContext* ctx) {
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  const Tensor& resource_handle_t = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
              errors::InvalidArgument("resource_handle must be a scalar"));
  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  IteratorVariantSerializer serializer;
  OP_REQUIRES_OK(ctx, serializer.InitializeFromIterator(
                          ctx, external_state_policy_, iterator_resource));
  Tensor* serialized_t;
  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(0, TensorShape({serializer.NumTensors()}),
                                      &serialized_t));
  OP_REQUIRES_OK(ctx, serializer.Serialize(serialized_t));
}

void DeserializeIteratorOp::Compute(OpKernelContext* ctx) {
  tensorflow::ResourceTagger tag(kTFDataResourceTag,
                                 ctx->op_kernel().type_string());
  // Validate that the handle corresponds to a real resource, and
  // that it is an IteratorResource.
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator_resource));
  core::ScopedUnref unref_iterator(iterator_resource);
  const Tensor* serialized_t;
  OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized_t));
  IteratorVariantSerializer serializer;
  OP_REQUIRES_OK(ctx, serializer.InitFromTensor(serialized_t));
  absl::Status s = iterator_resource->Restore(ctx, serializer.GetReader());
  if (!s.ok()) {
    OP_REQUIRES_OK(
        ctx,
        errors::CreateWithUpdatedMessage(
            s, absl::StrCat(
                   "Failed to restore dataset iterator from checkpoint: ",
                   s.message(),
                   ". Make sure the dataset definition has not changed between "
                   "the process that saved the checkpoint and the process that "
                   "is restoring it.")));
  }
}

namespace {

REGISTER_KERNEL_BUILDER(Name("Iterator").Device(DEVICE_CPU), IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorV2").Device(DEVICE_CPU).Priority(2),
                        IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorV2").Device(DEVICE_GPU).Priority(1),
                        IteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_CPU).Priority(2),
                        MakeIteratorOp);
REGISTER_KERNEL_BUILDER(
    Name("MakeIterator").Device(DEVICE_GPU).Priority(1).HostMemory("dataset"),
    MakeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("DeleteIterator").Device(DEVICE_CPU).Priority(2),
                        DeleteIteratorOp);
REGISTER_KERNEL_BUILDER(Name("DeleteIterator").Device(DEVICE_GPU).Priority(1),
                        DeleteIteratorOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIterator").Device(DEVICE_CPU).Priority(2),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIterator").Device(DEVICE_GPU).Priority(1),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV2").Device(DEVICE_CPU).Priority(2),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV2").Device(DEVICE_GPU).Priority(1),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV3").Device(DEVICE_CPU).Priority(2),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIteratorV3").Device(DEVICE_GPU).Priority(1),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("DatasetToSingleElement").Device(DEVICE_CPU),
                        ToSingleElementOp);
REGISTER_KERNEL_BUILDER(Name("OneShotIterator").Device(DEVICE_CPU),
                        OneShotIteratorOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_CPU).Priority(2),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_GPU).Priority(1),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextSync").Device(DEVICE_CPU).Priority(2),
    IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextSync").Device(DEVICE_GPU).Priority(1),
    IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextAsOptional").Device(DEVICE_CPU).Priority(2),
    IteratorGetNextAsOptionalOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextAsOptional").Device(DEVICE_GPU).Priority(1),
    IteratorGetNextAsOptionalOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorToStringHandle").Device(DEVICE_CPU).Priority(2),
    IteratorToStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorToStringHandle")
                            .Device(DEVICE_GPU)
                            .HostMemory("string_handle")
                            .Priority(1),
                        IteratorToStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorFromStringHandle").Device(DEVICE_CPU),
                        IteratorFromStringHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorFromStringHandleV2").Device(DEVICE_CPU).Priority(2),
    IteratorFromStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorFromStringHandleV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("string_handle")
                            .Priority(1),
                        IteratorFromStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("SerializeIterator").Device(DEVICE_CPU),
                        SerializeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("DeserializeIterator").Device(DEVICE_CPU),
                        DeserializeIteratorOp);

REGISTER_KERNEL_BUILDER(Name("IteratorGetModelProto").Device(DEVICE_CPU),
                        IteratorGetModelProtoOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
