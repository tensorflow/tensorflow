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

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/optional_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following ops.

const char kAnonymousIterator[] = "AnonymousIterator";
const char kAnonymousIteratorV2[] = "AnonymousIteratorV2";
const char kIteratorVariantTypeName[] = "tensorflow::Iterator";
const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";

}  // namespace

Status IteratorResource::GetNext(OpKernelContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) {
  std::shared_ptr<State> captured_state;
  {
    tf_shared_lock l(mu_);
    captured_state = iterator_state_;
  }
  if (captured_state->iterator) {
    IteratorContext::Params params(ctx);
    params.flr = captured_state->flr;
    params.function_handle_cache = captured_state->function_handle_cache.get();
    params.resource_mgr = &captured_state->resource_mgr;
    params.thread_factory = unbounded_thread_pool_.get_thread_factory();
    params.thread_pool = &unbounded_thread_pool_;
    params.cancellation_manager = &captured_state->cancellation_manager;
    std::function<void()> deregister_fn;
    TF_RETURN_IF_ERROR(RegisterCancellationCallback(
        ctx->cancellation_manager(),
        [cm = params.cancellation_manager]() { cm->StartCancel(); },
        &deregister_fn));
    auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
    auto val = captured_state->iterator->GetNext(
        IteratorContext(std::move(params)), out_tensors, end_of_sequence);
    metrics::RecordTFDataBytesFetched(GetTotalBytes(*out_tensors));
    return val;
  }
  return errors::FailedPrecondition(
      "GetNext() failed because the iterator has not been initialized. Ensure "
      "that you have run the initializer operation for this iterator before "
      "getting the next element.");
}

Status IteratorResource::Save(SerializationContext* ctx,
                              IteratorStateWriter* writer) {
  std::shared_ptr<State> captured_state;
  {
    tf_shared_lock l(mu_);
    captured_state = iterator_state_;
  }
  if (captured_state->iterator) {
    return captured_state->iterator->Save(ctx, writer);
  }
  return errors::FailedPrecondition(
      "Save() failed because the iterator has not been initialized. Ensure "
      "that you have run the initializer operation for this iterator before "
      "saving it.");
}

Status IteratorResource::Restore(OpKernelContext* ctx,
                                 IteratorStateReader* reader) {
  const DatasetBase* dataset;
  std::shared_ptr<State> new_state;
  {
    tf_shared_lock l(mu_);
    if (!iterator_state_->iterator) {
      return errors::FailedPrecondition(
          "Restore() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before restoring it.");
    }
    dataset = iterator_state_->iterator->dataset();
    // Hang onto a reference until we've created the new iterator, which will
    // then hold its own reference to keep the dataset alive.
    dataset->Ref();
    new_state = std::make_shared<State>(
        iterator_state_->flib_def, iterator_state_->pflr, iterator_state_->flr,
        /*iterator=*/nullptr);
  }
  core::ScopedUnref scoped_unref(dataset);
  IteratorContext::Params params(ctx);
  params.flr = new_state->flr;
  params.function_handle_cache = new_state->function_handle_cache.get();
  params.resource_mgr = &new_state->resource_mgr;
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.cancellation_manager = &new_state->cancellation_manager;
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
  std::unique_ptr<IteratorBase> iterator_base;
  TF_RETURN_IF_ERROR(dataset->MakeIteratorFromCheckpoint(
      IteratorContext(std::move(params)), "Iterator", reader, &iterator_base));
  new_state->DowncastAndSetIterator(std::move(iterator_base));

  mutex_lock l(mu_);
  std::swap(iterator_state_, new_state);
  return Status::OK();
}

Status IteratorResource::SetIteratorFromDataset(OpKernelContext* ctx,
                                                DatasetBase* dataset) {
  std::shared_ptr<State> new_state;
  {
    tf_shared_lock l(mu_);
    new_state = std::make_shared<State>(
        iterator_state_->flib_def, iterator_state_->pflr, iterator_state_->flr,
        /*iterator=*/nullptr);
  }
  // Create new iterator.
  std::unique_ptr<IteratorBase> iterator;
  IteratorContext::Params params(ctx);
  params.flr = new_state->flr;
  params.function_handle_cache = new_state->function_handle_cache.get();
  params.resource_mgr = &new_state->resource_mgr;
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  params.cancellation_manager = &new_state->cancellation_manager;
  std::function<void()> deregister_fn;
  TF_RETURN_IF_ERROR(RegisterCancellationCallback(
      ctx->cancellation_manager(),
      [cm = params.cancellation_manager]() { cm->StartCancel(); },
      &deregister_fn));
  {
    auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
    TF_RETURN_IF_ERROR(dataset->MakeIterator(IteratorContext(std::move(params)),
                                             "Iterator", &iterator));
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, iterator->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));

    new_state->DowncastAndSetIterator(std::move(iterator));
  }

  mutex_lock l(mu_);
  std::swap(iterator_state_, new_state);
  return Status::OK();
}

namespace {

// Wrapper for encoding/decoding the iterator state stored in a Variant tensor.
// The get() method returns an VariantTensorData object which contains all the
// state needed to restore a single iterator.
//
// Usage example:
//
// Encoding:
//
//   Tensor t(DT_VARIANT, TensorShape({}));
//   t->scalar<Variant>()() = IteratorStateVariant();
//
// Encode() sets the type_name of the VariantTensorData object to
// IteratorStateVariant::TypeName().
//
// Decoding:
//
//   Variant v = <VariantTensorDataProto object>;
//   DecodeUnaryVariant(&v);
//   IteratorStateVariant* wrapper = v.get<IteratorStateVariant>();
//   IteratorStateReader reader({wrapper->GetData()});
//   iterator_resource->Restore(ctx, &reader);
//
// The type_name of the VariantTensorData object to be decoded must
// match IteratorStateVariant::TypeName().
class IteratorStateVariant {
 public:
  IteratorStateVariant() : data_(nullptr) {}
  IteratorStateVariant(const IteratorStateVariant& other) : data_(nullptr) {
    if (other.data_) {
      Decode(*other.data_);
    }
  }
  IteratorStateVariant& operator=(IteratorStateVariant&& other) = default;
  IteratorStateVariant& operator=(const IteratorStateVariant& other) = delete;

  // Initializes `this` from a VariantTensorData object.
  Status InitializeFromVariantData(std::unique_ptr<VariantTensorData> d) {
    data_ = std::move(d);
    return Status::OK();
  }

  string TypeName() const { return kIteratorVariantTypeName; }
  void Encode(VariantTensorData* data) const { *data = *data_; }
  bool Decode(VariantTensorData data) {
    if (data.type_name() != TypeName()) {
      return false;
    }
    auto tensor_data = absl::make_unique<VariantTensorData>();
    std::swap(*tensor_data, data);
    data_ = std::move(tensor_data);
    return true;
  }

  // Returns a borrowed pointer to the underlying VariantTensorData.
  const VariantTensorData* GetData() const { return data_.get(); }

  string DebugString() const {
    if (data_) {
      return strings::StrCat("IteratorStateVariant<", data_->DebugString(),
                             ">");
    } else {
      return strings::StrCat("IteratorStateVariant<empty>");
    }
  }

 private:
  std::unique_ptr<VariantTensorData> data_;
};

// Register the reader class in the global variant decode_fn registry
// so that a Variant containing a serialized representation of iterator state
// can be decoded using DecodeUnaryVariant. If we don't do this we will need
// to manually decode the returned Variant using MaybeDecodeAndCopy in
// DeserializeIteratorOp which is not recommended.
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(IteratorStateVariant,
                                       kIteratorVariantTypeName);

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
  IteratorVariantSerializer() {}

  // Calls `Save` on the iterator_resource to build up the list of
  // IteratorStateVariant objects.
  Status InitializeFromIterator(IteratorResource* iterator_resource) {
    SerializationContext serialization_ctx({});
    VariantTensorDataWriter writer;
    TF_RETURN_IF_ERROR(iterator_resource->Save(&serialization_ctx, &writer));
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
    return Status::OK();
  }

  // Initializes `this` from `serialized_t` while restoring the iterator state.
  Status InitFromTensor(const Tensor* serialized_t) {
    int64 num_tensors = serialized_t->dim_size(0);
    auto serialized_vec = serialized_t->vec<Variant>();
    std::vector<const VariantTensorData*> data;
    data.reserve(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
      auto* w = serialized_vec(i).get<IteratorStateVariant>();
      data.push_back(w->GetData());
    }
    reader_ = absl::make_unique<VariantTensorDataReader>(data);
    num_tensors_ = data.size();
    return Status::OK();
  }

  int64 NumTensors() { return num_tensors_; }

  // Stores the IteratorStateVariant list into a pre-allocated tensor. Expects
  // that InitializeFromIterator was called before.
  Status Serialize(Tensor* serialized) {
    if (!can_serialize_) {
      return errors::InvalidArgument(
          "Please call InitializeFromIterator before calling Serialize.");
    }
    int64 size = variants_.size();
    for (int64 i = 0; i < size; ++i) {
      serialized->vec<Variant>()(i) = variants_[i];
    }
    return Status::OK();
  }

  // Returns an IteratorStateReader to restore iterator state. Expects that
  // InitFromTensor was called before.
  IteratorStateReader* GetReader() { return reader_.get(); }

 private:
  bool can_serialize_ = false;
  int64 num_tensors_;
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

void IteratorHandleOp::Compute(OpKernelContext* context) LOCKS_EXCLUDED(mu_) {
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
               this](IteratorResource** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                *ret = new IteratorResource(
                    context->env(), output_dtypes_, output_shapes_,
                    graph_def_version_, std::move(device_mgr),
                    std::move(flib_def), std::move(pflr), flr);
                return Status::OK();
              }));

      Status s = VerifyResource(resource);
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
                              MakeTypeIndex<IteratorResource>()));
}

Status IteratorHandleOp::VerifyResource(IteratorResource* resource) {
  TF_RETURN_IF_ERROR(
      VerifyTypesMatch(output_dtypes_, resource->output_dtypes()));
  TF_RETURN_IF_ERROR(
      VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
  return Status::OK();
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
      absl::make_unique<StaticDeviceMgr>(RenamedDevice::NewRenamedDevice(
          ctx->device()->name(), down_cast<Device*>(ctx->device()),
          false /* owns_underlying */, false /* isolate_session_state */));
  *flib_def = absl::make_unique<FunctionLibraryDefinition>(
      *ctx->function_library()->GetFunctionLibraryDefinition());
  const auto* config = ctx->function_library()->config_proto();
  *pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
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
    : AnonymousResourceOp<IteratorResource>(context),
      graph_def_version_(context->graph_def_version()) {
  OP_REQUIRES_OK(context, context->GetAttr(kOutputTypes, &output_dtypes_));
  OP_REQUIRES_OK(context, context->GetAttr(kOutputShapes, &output_shapes_));
  create_deleter_ = context->def().op() == kAnonymousIteratorV2;
}

string AnonymousIteratorHandleOp::name() { return kAnonymousIterator; }

Status AnonymousIteratorHandleOp::CreateResource(
    OpKernelContext* ctx, std::unique_ptr<FunctionLibraryDefinition> flib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
    FunctionLibraryRuntime* lib, IteratorResource** resource) {
  std::unique_ptr<DeviceMgr> device_mgr(nullptr);
  *resource = new IteratorResource(ctx->env(), output_dtypes_, output_shapes_,
                                   graph_def_version_, std::move(device_mgr),
                                   std::move(flib_def), std::move(pflr), lib);
  return Status::OK();
}

void MakeIteratorOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  DatasetBase* dataset;
  OP_REQUIRES_OK_ASYNC(
      ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset), done);
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK_ASYNC(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource),
      done);
  background_worker_.Schedule(std::bind(
      [ctx, iterator_resource, dataset](DoneCallback done) {
        Status s = iterator_resource->SetIteratorFromDataset(ctx, dataset);
        iterator_resource->Unref();
        if (!s.ok()) {
          ctx->SetStatus(s);
        }
        done();
      },
      std::move(done)));
}

void DeleteIteratorOp::Compute(OpKernelContext* ctx) {
  ResourceHandle handle = ctx->input(0).flat<ResourceHandle>()(0);
  // The iterator resource is guaranteed to exist because the variant tensor
  // wrapping the deleter is provided as an unused input to this op, which
  // guarantees that it has not run yet.
  OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete(handle));
}

namespace {

class ToSingleElementOp : public AsyncOpKernel {
 public:
  explicit ToSingleElementOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_to_single_element") {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    background_worker_.Schedule([this, ctx, done = std::move(done)]() {
      OP_REQUIRES_OK_ASYNC(ctx, DoCompute(ctx), done);
      done();
    });
  }

 private:
  Status DoCompute(OpKernelContext* ctx) {
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    IteratorContext::Params params(ctx);
    FunctionHandleCache function_handle_cache(params.flr);
    params.function_handle_cache = &function_handle_cache;
    ResourceMgr resource_mgr;
    params.resource_mgr = &resource_mgr;
    CancellationManager cancellation_manager(ctx->cancellation_manager());
    params.cancellation_manager = &cancellation_manager;

    IteratorContext iter_ctx(std::move(params));
    std::unique_ptr<IteratorBase> iterator;
    TF_RETURN_IF_ERROR(
        dataset->MakeIterator(&iter_ctx, "SingleElementIterator", &iterator));

    std::vector<Tensor> components;
    components.reserve(dataset->output_dtypes().size());
    bool end_of_sequence = false;

    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &components, &end_of_sequence));

    if (end_of_sequence) {
      return errors::InvalidArgument("Dataset was empty.");
    }
    for (int i = 0; i < components.size(); ++i) {
      // TODO(mrry): Check that the shapes match the shape attrs.
      ctx->set_output(i, components[i]);
    }

    components.clear();
    TF_RETURN_IF_ERROR(
        iterator->GetNext(&iter_ctx, &components, &end_of_sequence));
    if (!end_of_sequence) {
      return errors::InvalidArgument("Dataset had more than one element.");
    }
    return Status::OK();
  }

  BackgroundWorker background_worker_;
};

class ReduceDatasetOp : public AsyncOpKernel {
 public:
  explicit ReduceDatasetOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_reduce_dataset") {
    FunctionMetadata::Params params;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_inter_op_parallelism",
                                     &params.use_inter_op_parallelism));
    params.is_multi_device_function = true;
    OP_REQUIRES_OK(ctx,
                   FunctionMetadata::Create(ctx, "f", params, &func_metadata_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    background_worker_.Schedule([this, ctx, done = std::move(done)]() {
      OP_REQUIRES_OK_ASYNC(ctx, DoCompute(ctx), done);
      done();
    });
  }

 private:
  Status DoCompute(OpKernelContext* ctx) {
    DatasetBase* dataset;
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));
    OpInputList inputs;
    TF_RETURN_IF_ERROR(ctx->input_list("initial_state", &inputs));
    std::vector<Tensor> state(inputs.begin(), inputs.end());

    std::unique_ptr<CapturedFunction> captured_func;
    TF_RETURN_IF_ERROR(CapturedFunction::Create(
        ctx, func_metadata_, "other_arguments", &captured_func));

    IteratorContext::Params params(ctx);
    auto function_handle_cache =
        absl::make_unique<FunctionHandleCache>(params.flr);
    params.function_handle_cache = function_handle_cache.get();
    ResourceMgr resource_mgr;
    params.resource_mgr = &resource_mgr;
    CancellationManager cancellation_manager(ctx->cancellation_manager());
    params.cancellation_manager = &cancellation_manager;

    IteratorContext iter_ctx(std::move(params));
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func;
    TF_RETURN_IF_ERROR(
        captured_func->Instantiate(&iter_ctx, &instantiated_captured_func));

    std::unique_ptr<IteratorBase> iterator;
    TF_RETURN_IF_ERROR(
        dataset->MakeIterator(&iter_ctx, "ReduceIterator", &iterator));

    // Iterate through the input dataset.
    while (true) {
      if (ctx->cancellation_manager()->IsCancelled()) {
        return errors::Cancelled("Operation was cancelled");
      }
      std::vector<Tensor> next_input_element;
      bool end_of_input;
      TF_RETURN_IF_ERROR(
          iterator->GetNext(&iter_ctx, &next_input_element, &end_of_input));
      if (end_of_input) {
        break;
      }

      // Run the reduce function to update the current state.
      std::vector<Tensor> args;
      args.reserve(state.size() + next_input_element.size());
      std::copy(state.begin(), state.end(), std::back_inserter(args));
      std::copy(next_input_element.begin(), next_input_element.end(),
                std::back_inserter(args));

      std::vector<Tensor> reduce_func_output;
      TF_RETURN_IF_ERROR(instantiated_captured_func->Run(
          &iter_ctx, std::move(args), &reduce_func_output));
      if (reduce_func_output.size() != state.size()) {
        return errors::InvalidArgument(
            "The number of components of the initial state and the "
            "reduce "
            "function output does not match. (initial_state=",
            state.size(), ", output=", reduce_func_output.size(), ").");
      }
      std::swap(reduce_func_output, state);
    }

    if (state.size() != output_types_.size()) {
      return errors::InvalidArgument(
          "The number of result elements does not match "
          "the size of output types: ",
          state.size(), " vs. ", output_types_.size());
    }
    if (state.size() != output_shapes_.size()) {
      return errors::InvalidArgument(
          "The number of result elements does not match "
          "the size of output shapes: ",
          state.size(), " vs. ", output_shapes_.size());
    }
    for (size_t i = 0; i < state.size(); ++i) {
      if (state[i].dtype() != output_types_[i]) {
        return errors::InvalidArgument(
            "The result does not match the expected type for "
            "component ",
            i, ". Expected: ", DataTypeString(output_types_[i]),
            ". Actual: ", DataTypeString(state[i].dtype()), ".");
      }
      if (!output_shapes_[i].IsCompatibleWith(state[i].shape())) {
        return errors::InvalidArgument(
            "The result does not match the expected shape for "
            "component ",
            i, ". Expected: ", output_shapes_[i].DebugString(),
            ". Actual: ", state[i].shape().DebugString(), ".");
      }
      ctx->set_output(i, state[i]);
    }
    return Status::OK();
  }

  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  BackgroundWorker background_worker_;
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
    Status s = TryInit(ctx, &iterator, &cinfo);

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

  Status TryInit(OpKernelContext* ctx, IteratorResource** iterator,
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
                EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                  *ret = new IteratorResource(
                      ctx->env(), output_dtypes_, output_shapes_,
                      graph_def_version_, nullptr, std::move(flib_def),
                      std::move(pflr), flr);
                  return Status::OK();
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
    Notification n;
    Status factory_status;
    std::vector<Tensor> return_values;
    ctx->function_library()->Run(opts, f_handle, {}, &return_values,
                                 [&n, &factory_status](Status s) {
                                   factory_status.Update(s);
                                   n.Notify();
                                 });
    n.WaitForNotification();
    TF_RETURN_IF_ERROR(factory_status);
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
    return Status::OK();
  }

  void ProduceOutput(OpKernelContext* ctx, const DoneCallback& done) {
    Tensor* handle;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, TensorShape({}), &handle),
                         done);
    Status s;
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
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  IteratorResource* iterator_resource_ GUARDED_BY(mu_) = nullptr;

  bool initialization_started_ GUARDED_BY(mu_) = false;
  Status initialization_status_ GUARDED_BY(mu_);
  std::vector<std::pair<OpKernelContext*, DoneCallback>> done_callbacks_
      GUARDED_BY(mu_);
  const int graph_def_version_;
};

}  // namespace

void IteratorGetNextOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  IteratorResource* iterator;
  OP_REQUIRES_OK_ASYNC(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator), done);
  // The call to `iterator->GetNext()` may block and depend on an
  // inter-op thread pool thread, so we issue the call from the
  // owned thread pool.
  background_worker_.Schedule(std::bind(
      [ctx, iterator](DoneCallback done) {
        std::vector<Tensor> components;
        bool end_of_sequence = false;

        Status s = iterator->GetNext(ctx, &components, &end_of_sequence);
        // NOTE(mrry): We must unref the iterator before calling `done()`, to
        // avoid destruction races.
        iterator->Unref();

        if (!s.ok()) {
          ctx->SetStatus(s);
        } else if (end_of_sequence) {
          ctx->SetStatus(errors::OutOfRange("End of sequence"));
        } else {
          for (int i = 0; i < components.size(); ++i) {
            // TODO(mrry): Check that the shapes match the shape attrs.
            ctx->set_output(i, components[i]);
          }
        }
        done();
      },
      std::move(done)));
}

void IteratorGetNextSyncOp::Compute(OpKernelContext* ctx) {
  IteratorResource* iterator;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
  core::ScopedUnref unref_iterator(iterator);
  std::vector<Tensor> components;
  bool end_of_sequence = false;

  OP_REQUIRES_OK(ctx, iterator->GetNext(ctx, &components, &end_of_sequence));
  OP_REQUIRES(ctx, !end_of_sequence, errors::OutOfRange("End of sequence"));

  for (int i = 0; i < components.size(); ++i) {
    // TODO(mrry): Check that the shapes match the shape attrs.
    ctx->set_output(i, components[i]);
  }
}

void IteratorGetNextAsOptionalOp::ComputeAsync(OpKernelContext* ctx,
                                               DoneCallback done) {
  IteratorResource* iterator;
  OP_REQUIRES_OK_ASYNC(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator), done);
  // The call to `iterator->GetNext()` may block and depend on an
  // inter-op thread pool thread, so we issue the call from the
  // owned thread pool.
  background_worker_.Schedule(std::bind(
      [this, ctx, iterator](DoneCallback done) {
        std::vector<Tensor> components;
        bool end_of_sequence = false;

        Status s = iterator->GetNext(ctx, &components, &end_of_sequence);
        // NOTE(mrry): We must unref the iterator before calling `done()`, to
        // avoid destruction races.
        iterator->Unref();

        if (!s.ok()) {
          ctx->SetStatus(s);
        } else if (end_of_sequence) {
          OP_REQUIRES_OK_ASYNC(ctx, WriteOptionalNoneToOutput(ctx, 0), done);
        } else {
          for (int i = 0; i < components.size(); ++i) {
            OP_REQUIRES_ASYNC(
                ctx, components[i].dtype() == output_types_[i],
                errors::InvalidArgument(
                    "The given optional does not match the expected type for "
                    "component ",
                    i, ". Expected: ", DataTypeString(output_types_[i]),
                    ". Actual: ", DataTypeString(components[i].dtype()), "."),
                done);
            OP_REQUIRES_ASYNC(
                ctx, output_shapes_[i].IsCompatibleWith(components[i].shape()),
                errors::InvalidArgument(
                    "The given optional does not match the expected shape "
                    "for component ",
                    i, ". Expected: ", output_shapes_[i].DebugString(),
                    ". Actual: ", components[i].shape().DebugString(), "."),
                done);
          }

          OP_REQUIRES_OK_ASYNC(
              ctx,
              WriteOptionalWithValueToOutput(ctx, 0, std::move(components)),
              done);
        }
        done();
      },
      std::move(done)));
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

namespace {

class SerializeIteratorOp : public OpKernel {
 public:
  explicit SerializeIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
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
    OP_REQUIRES_OK(ctx, serializer.InitializeFromIterator(iterator_resource));
    Tensor* serialized_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({serializer.NumTensors()}),
                                  &serialized_t));
    OP_REQUIRES_OK(ctx, serializer.Serialize(serialized_t));
  }
};

class DeserializeIteratorOp : public OpKernel {
 public:
  explicit DeserializeIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
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
    OP_REQUIRES_OK(ctx,
                   iterator_resource->Restore(ctx, serializer.GetReader()));
  }
};

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
REGISTER_KERNEL_BUILDER(
    Name("DeleteIterator").Device(DEVICE_GPU).HostMemory("deleter").Priority(1),
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
REGISTER_KERNEL_BUILDER(Name("AnonymousIteratorV2")
                            .Device(DEVICE_GPU)
                            .HostMemory("deleter")
                            .Priority(1),
                        AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("DatasetToSingleElement").Device(DEVICE_CPU),
                        ToSingleElementOp);
REGISTER_KERNEL_BUILDER(Name("ReduceDataset").Device(DEVICE_CPU),
                        ReduceDatasetOp);
REGISTER_KERNEL_BUILDER(Name("OneShotIterator").Device(DEVICE_CPU),
                        OneShotIteratorOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_CPU).Priority(2),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_GPU).Priority(1),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextSync").Device(DEVICE_CPU).Priority(2),
    IteratorGetNextSyncOp);
REGISTER_KERNEL_BUILDER(
    Name("IteratorGetNextSync").Device(DEVICE_GPU).Priority(1),
    IteratorGetNextSyncOp);
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

REGISTER_INPUT_COLOCATION_EXEMPTION("ReduceDataset");

}  // namespace

}  // namespace data
}  // namespace tensorflow
