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

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/optional_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
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

const char kIteratorVariantTypeName[] = "tensorflow::Iterator";

}  // namespace

class IteratorResource : public ResourceBase {
 public:
  IteratorResource(const DataTypeVector& output_dtypes,
                   const std::vector<PartialTensorShape>& output_shapes,
                   const int /*unused: graph_def_version*/,
                   std::unique_ptr<DeviceMgr> device_mgr,
                   std::unique_ptr<FunctionLibraryDefinition> flib_def,
                   std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                   FunctionLibraryRuntime* lib)
      : device_mgr_(std::move(device_mgr)),
        iterator_state_(std::make_shared<State>(
            std::move(flib_def), std::move(pflr), lib, nullptr /* iterator */)),
        output_dtypes_(output_dtypes),
        output_shapes_(output_shapes) {}

  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
    std::shared_ptr<State> captured_state;
    {
      tf_shared_lock l(mu_);
      captured_state = iterator_state_;
    }
    if (captured_state->iterator) {
      IteratorContext::Params params(ctx);
      params.lib = captured_state->lib;
      params.function_handle_cache =
          captured_state->function_handle_cache.get();
      params.resource_mgr = &captured_state->resource_mgr;
      return captured_state->iterator->GetNext(
          IteratorContext(std::move(params)), out_tensors, end_of_sequence);
    } else {
      return errors::FailedPrecondition(
          "GetNext() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before getting the next element.");
    }
  }

  Status GetNext(IteratorContext&& ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
    return GetNext(&ctx, out_tensors, end_of_sequence);
  }

  Status Save(SerializationContext* ctx, IteratorStateWriter* writer) {
    std::shared_ptr<State> captured_state;
    {
      tf_shared_lock l(mu_);
      captured_state = iterator_state_;
    }
    if (captured_state) {
      return captured_state->iterator->Save(ctx, writer);
    } else {
      return errors::FailedPrecondition(
          "Save() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before saving it.");
    }
  }

  Status Restore(OpKernelContext* ctx, IteratorStateReader* reader) {
    string serialized_graph_def;
    TF_RETURN_IF_ERROR(reader->ReadScalar(DatasetBase::kDatasetGraphKey,
                                          &serialized_graph_def));
    GraphDef graph_def;
    if (!graph_def.ParseFromString(serialized_graph_def)) {
      return errors::Internal("Error parsing dataset GraphDef.");
    }
    string output_node;
    TF_RETURN_IF_ERROR(reader->ReadScalar(
        DatasetBase::kDatasetGraphOutputNodeKey, &output_node));
    DatasetBase* dataset = nullptr;
    Graph graph(OpRegistry::Global());
    TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, nullptr));
    std::vector<Tensor> outputs;
    GraphRunner graph_runner(ctx->env());

    // Build a new FLR that knows about the functions in the graph, and use
    // it for all operations on the restored iterator.
    // NOTE(mrry): We clone the existing FLR and use it in the GraphRunner
    // because some of the OpKernels in the graph might call functions that are
    // only defined in the loaded GraphDef.
    FunctionLibraryRuntime* lib;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    TF_RETURN_IF_ERROR(ctx->function_library()->Clone(&flib_def, &pflr, &lib));
    TF_RETURN_IF_ERROR(flib_def->AddLibrary(graph_def.library()));
    std::unique_ptr<State> new_state = absl::make_unique<State>(
        std::move(flib_def), std::move(pflr), lib, nullptr /* iterator */);

    TF_RETURN_IF_ERROR(
        graph_runner.Run(&graph, new_state->lib, {}, {output_node}, &outputs));
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], &dataset));

    IteratorContext::Params params(ctx);
    params.lib = new_state->lib;
    params.function_handle_cache = new_state->function_handle_cache.get();
    params.resource_mgr = &new_state->resource_mgr;
    TF_RETURN_IF_ERROR(dataset->MakeIterator(IteratorContext(std::move(params)),
                                             "Iterator", &new_state->iterator));
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, new_state->iterator->output_dtypes()));
    TF_RETURN_IF_ERROR(VerifyShapesCompatible(
        output_shapes_, new_state->iterator->output_shapes()));

    {
      IteratorContext::Params params(ctx);
      params.lib = new_state->lib;
      params.function_handle_cache = new_state->function_handle_cache.get();
      params.resource_mgr = &new_state->resource_mgr;
      DeviceBase* device = new_state->lib->device();
      params.allocator_getter = [device](AllocatorAttributes attrs) {
        return device->GetAllocator(attrs);
      };
      IteratorContext iter_ctx(std::move(params));
      TF_RETURN_IF_ERROR(new_state->iterator->Restore(&iter_ctx, reader));
    }

    mutex_lock l(mu_);
    iterator_state_ = std::move(new_state);
    return Status::OK();
  }

  Status AddLibrary(const FunctionLibraryDefinition& flib_def) {
    mutex_lock l(mu_);
    return iterator_state_->flib_def->AddLibrary(flib_def);
  }

  Status SetIteratorFromDataset(OpKernelContext* ctx, DatasetBase* dataset) {
    std::shared_ptr<State> new_state;
    {
      tf_shared_lock l(mu_);
      new_state = std::make_shared<State>(
          iterator_state_->flib_def, iterator_state_->pflr,
          iterator_state_->lib, nullptr /* function_handle_cache */,
          nullptr /* iterator */);
    }

    // Ensure that the iterator has access to all functions in the current
    // subgraph, because some functions may have been defined after the resource
    // was initially created.
    Status s = new_state->flib_def->AddLibrary(
        *ctx->function_library()->GetFunctionLibraryDefinition());

    if (!s.ok()) {
      // Adding functions to `flib_def_` may fail, if there are clashes between
      // the function names in (e.g.) a restored graph and the currently
      // executing graph. In that case, we create a new function runtime for
      // this iterator, based on the current `OpKernelContext`, which will have
      // the functions we need.
      FunctionLibraryRuntime* lib;
      std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
      TF_RETURN_IF_ERROR(
          ctx->function_library()->Clone(&flib_def, &pflr, &lib));
      new_state->flib_def = std::move(flib_def);
      new_state->pflr = std::move(pflr);
      new_state->lib = lib;
    }

    new_state->function_handle_cache =
        absl::make_unique<FunctionHandleCache>(new_state->lib);
    // Create new iterator.
    std::unique_ptr<IteratorBase> iterator;
    IteratorContext::Params params(ctx);
    params.lib = new_state->lib;
    params.function_handle_cache = new_state->function_handle_cache.get();
    params.resource_mgr = &new_state->resource_mgr;
    TF_RETURN_IF_ERROR(dataset->MakeIterator(IteratorContext(std::move(params)),
                                             "Iterator", &iterator));
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, iterator->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));
    std::swap(new_state->iterator, iterator);

    mutex_lock l(mu_);
    std::swap(iterator_state_, new_state);
    return Status::OK();
  }

  string DebugString() const override { return "Iterator resource"; }

  const DataTypeVector& output_dtypes() const { return output_dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

 private:
  struct State {
    State(std::shared_ptr<FunctionLibraryDefinition> flib_def,
          std::shared_ptr<ProcessFunctionLibraryRuntime> pflr,
          FunctionLibraryRuntime* lib, std::unique_ptr<IteratorBase> iterator)
        : flib_def(flib_def),
          pflr(pflr),
          lib(lib),
          function_handle_cache(absl::make_unique<FunctionHandleCache>(lib)),
          iterator(std::move(iterator)) {}

    State(std::shared_ptr<FunctionLibraryDefinition> flib_def,
          std::shared_ptr<ProcessFunctionLibraryRuntime> pflr,
          FunctionLibraryRuntime* lib,
          std::unique_ptr<FunctionHandleCache> function_handle_cache,
          std::unique_ptr<IteratorBase> iterator)
        : flib_def(flib_def),
          pflr(pflr),
          lib(lib),
          function_handle_cache(std::move(function_handle_cache)),
          iterator(std::move(iterator)) {}

    std::shared_ptr<FunctionLibraryDefinition> flib_def;
    std::shared_ptr<ProcessFunctionLibraryRuntime> pflr;
    FunctionLibraryRuntime* lib = nullptr;  // not owned.
    std::unique_ptr<FunctionHandleCache> function_handle_cache;
    ResourceMgr resource_mgr;
    std::unique_ptr<IteratorBase> iterator;
  };

  mutex mu_;
  const std::unique_ptr<DeviceMgr> device_mgr_ GUARDED_BY(mu_);
  std::shared_ptr<State> iterator_state_ GUARDED_BY(mu_);
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
};

namespace {

constexpr char kDelimiter[] = "@@";

// Helper class for reading data from a VariantTensorData object.
class VariantTensorDataReader : public IteratorStateReader {
 public:
  explicit VariantTensorDataReader(const VariantTensorData* data)
      : data_(data) {
    string metadata;
    data_->get_metadata(&metadata);
    auto keys = str_util::Split(metadata, kDelimiter, str_util::SkipEmpty());
    for (size_t i = 0; i < keys.size(); ++i) {
      map_[keys[i]] = i;
    }
  }

  // Returns OK iff the initialization was successful, i.e.,
  // pre-processing did not have errors.
  Status status() const { return status_; }

  Status ReadScalar(StringPiece key, int64* val) override {
    return ReadScalarInternal(key, val);
  }

  Status ReadScalar(StringPiece key, string* val) override {
    return ReadScalarInternal(key, val);
  }

  Status ReadTensor(StringPiece key, Tensor* val) override {
    return ReadTensorInternal(key, val);
  }

  bool Contains(StringPiece key) override {
    return map_.find(string(key)) != map_.end();
  }

 private:
  template <typename T>
  Status ReadScalarInternal(StringPiece key, T* val) {
    if (map_.find(string(key)) == map_.end()) {
      return errors::NotFound(key);
    }
    *val = data_->tensors(map_[string(key)]).scalar<T>()();
    return Status::OK();
  }

  Status ReadTensorInternal(StringPiece key, Tensor* val) {
    if (map_.find(string(key)) == map_.end()) {
      return errors::NotFound(key);
    }
    *val = data_->tensors(map_[string(key)]);
    return Status::OK();
  }

  std::map<string, size_t> map_;
  const VariantTensorData* data_;  // Not owned.
  Status status_;
};

// Helper class for writing data to a VariantTensorData object.
class VariantTensorDataWriter : public IteratorStateWriter {
 public:
  // Does not take ownership of data.
  explicit VariantTensorDataWriter(VariantTensorData* data) : data_(data) {}

  Status WriteScalar(StringPiece key, const int64 val) override {
    return WriteScalarInternal(key, val);
  }

  Status WriteScalar(StringPiece key, const string& val) override {
    return WriteScalarInternal(key, val);
  }

  Status WriteTensor(StringPiece key, const Tensor& val) override {
    return WriteTensorInternal(key, val);
  }

  Status Flush() {
    string metadata;
    for (size_t i = 0; i < keys_.size(); ++i) {
      strings::StrAppend(&metadata, kDelimiter, keys_[i]);
    }
    data_->set_metadata(metadata);
    return Status::OK();
  }

 private:
  template <typename T>
  Status WriteScalarInternal(StringPiece key, const T& val) {
    Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
    val_t.scalar<T>()() = val;
    return WriteTensorInternal(key, val_t);
  }

  Status WriteTensorInternal(StringPiece key, const Tensor& val) {
    DCHECK_EQ(key.find(kDelimiter), string::npos);
    keys_.push_back(string(key));
    *(data_->add_tensors()) = val;
    return Status::OK();
  }

  VariantTensorData* data_;
  std::vector<string> keys_;
};

// Wrapper for encoding/decoding the iterator state stored in a Variant tensor.
// The get() method returns an IteratorStateReader which can be used
// to restore iterator state.
//
// Usage example:
//
// Encoding:
//
//   Tensor t(DT_VARIANT, TensorShape({}));
//   t->scalar<Variant>()() = IteratorStateVariant(iterator_resource);
//
// Encode() sets the type_name of the VariantTensorData object to
// IteratorStateVariant::TypeName().
//
// Decoding:
//
//   Variant v = <VariantTensorDataProto object>;
//   DecodeUnaryVariant(&v);
//   IteratorStateVariant* wrapper = v.get<IteratorStateVariant>();
//   iterator_resource->Restore(ctx, wrapper->get())
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
  // Initializes this object with the current state of the iterator so
  // that it can be written on the next call to Encode().
  Status InitializeFromIterator(OpKernelContext* ctx,
                                IteratorResource* iterator_resource) {
    SerializationContext::Params params;
    params.flib_def = ctx->function_library()->GetFunctionLibraryDefinition();
    SerializationContext serialization_ctx(params);
    data_ = absl::make_unique<VariantTensorData>();
    data_->set_type_name(TypeName());
    VariantTensorDataWriter writer(data_.get());
    TF_RETURN_IF_ERROR(iterator_resource->Save(&serialization_ctx, &writer));
    TF_RETURN_IF_ERROR(writer.Flush());
    return Status::OK();
  }
  string TypeName() const { return kIteratorVariantTypeName; }
  void Encode(VariantTensorData* data) const { *data = *data_; }
  bool Decode(VariantTensorData data) {
    if (data.type_name() != TypeName()) {
      return false;
    }
    std::unique_ptr<VariantTensorData> tensor_data =
        absl::make_unique<VariantTensorData>();
    std::swap(*tensor_data, data);
    std::unique_ptr<VariantTensorDataReader> reader =
        absl::make_unique<VariantTensorDataReader>(tensor_data.get());
    status_ = reader->status();
    if (!status_.ok()) {
      return false;
    }
    data_ = std::move(tensor_data);
    reader_ = std::move(reader);
    return true;
  }
  IteratorStateReader* get() { return reader_.get(); }
  Status status() const { return status_; }
  string DebugString() const {
    if (data_) {
      return strings::StrCat("IteratorStateVariant<",
                             "data: ", data_->DebugString(),
                             " status: ", status_.ToString(), ">");
    } else {
      return strings::StrCat("IteratorStateVariant<empty>");
    }
  }

 private:
  std::unique_ptr<IteratorStateReader> reader_;
  Status status_;
  std::unique_ptr<VariantTensorData> data_;
};

// Register the reader class in the global variant decode_fn registry
// so that a Variant containing a serialized representation of iterator state
// can be decoded using DecodeUnaryVariant. If we don't do this we will need
// to manually decode the returned Variant using MaybeDecodeAndCopy in
// DeserializeIteratorOp which is not recommended.
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(IteratorStateVariant,
                                       kIteratorVariantTypeName);

}  // namespace

// Note that IteratorHandleOp holds a reference to the resource it creates. If
// cleaning up resources with DestroyResourceOp is important, consider creating
// resource containers with AnonymousIteratorHandleOp instead.
IteratorHandleOp::IteratorHandleOp(OpKernelConstruction* ctx)
    : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
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
      FunctionLibraryRuntime* lib;
      std::unique_ptr<DeviceMgr> device_mgr(nullptr);
      std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
      // If the iterator is shared then we construct a new FLR, and pass that
      // in. NOTE(mrry,rohanj): In this case it is not possible to call remote
      // functions from the iterator. We may add this functionality if there
      // is sufficient demand, but it will require a significant refactoring.
      if (!name_.empty()) {
        lib = CreatePrivateFLR(context, &device_mgr, &flib_def, &pflr);
      } else {
        OP_REQUIRES_OK(context, context->function_library()->Clone(
                                    &flib_def, &pflr, &lib));
      }

      ResourceMgr* mgr = context->resource_manager();
      OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

      IteratorResource* resource;
      OP_REQUIRES_OK(
          context,
          mgr->LookupOrCreate<IteratorResource>(
              cinfo_.container(), cinfo_.name(), &resource,
              [lib, &device_mgr, &flib_def, &pflr, this](IteratorResource** ret)
                  EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                    *ret = new IteratorResource(
                        output_dtypes_, output_shapes_, graph_def_version_,
                        std::move(device_mgr), std::move(flib_def),
                        std::move(pflr), lib);
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
  *device_mgr = absl::make_unique<DeviceMgr>(RenamedDevice::NewRenamedDevice(
      ctx->device()->name(), down_cast<Device*>(ctx->device()),
      false /* owns_underlying */, false /* isolate_session_state */));
  *flib_def = absl::make_unique<FunctionLibraryDefinition>(
      *ctx->function_library()->GetFunctionLibraryDefinition());
  *pflr = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr->get(), ctx->env(), graph_def_version_, flib_def->get(),
      OptimizerOptions{} /* TODO(mrry): OptimizerOptions? */,
      nullptr /* TODO(mrry): ClusterFLR */);

  return (*pflr)->GetFLR(ctx->device()->name());
}

// Like IteratorHandleOp, but creates handles which are never shared, and does
// not hold a reference to these handles. The latter is important for eager
// execution, since OpKernel instances generally live as long as the program
// running them.
AnonymousIteratorHandleOp::AnonymousIteratorHandleOp(
    OpKernelConstruction* context)
    : OpKernel(context), graph_def_version_(context->graph_def_version()) {
  OP_REQUIRES_OK(context, context->GetAttr("output_types", &output_dtypes_));
  OP_REQUIRES_OK(context, context->GetAttr("output_shapes", &output_shapes_));
}

void AnonymousIteratorHandleOp::Compute(OpKernelContext* context) {
  FunctionLibraryRuntime* lib;
  std::unique_ptr<DeviceMgr> device_mgr(nullptr);
  std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
  OP_REQUIRES_OK(context,
                 context->function_library()->Clone(&flib_def, &pflr, &lib));

  ResourceMgr* mgr = context->resource_manager();

  const string container_name = "AnonymousIterator";
  string unique_name;
  {
    mutex_lock l(static_resource_lookup_mutex_);
    while (true) {  // Find an unused name
      IteratorResource* existing_resource = nullptr;
      unique_name = strings::StrCat("AnonymousIterator", current_id_++);
      Status status = mgr->Lookup<IteratorResource>(container_name, unique_name,
                                                    &existing_resource);
      if (status.code() == error::NOT_FOUND) {
        break;
      }
      OP_REQUIRES_OK(context, status);
      existing_resource->Unref();
    }
    IteratorResource* new_resource = new IteratorResource(
        output_dtypes_, output_shapes_, graph_def_version_,
        std::move(device_mgr), std::move(flib_def), std::move(pflr), lib);
    // Create the resource with our chosen name under the resource lookup
    // mutex to avoid another kernel racily creating a resource with this
    // name.
    OP_REQUIRES_OK(context, mgr->Create<IteratorResource>(
                                container_name, unique_name, new_resource));
  }
  OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                              context, 0, container_name, unique_name,
                              MakeTypeIndex<IteratorResource>()));
}

// Static initializers for AnonymousIteratorHandleOp id counting.
mutex AnonymousIteratorHandleOp::static_resource_lookup_mutex_{
    LINKER_INITIALIZED};
int64 AnonymousIteratorHandleOp::current_id_(0);

void MakeIteratorOp::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
  IteratorResource* iterator_resource;
  OP_REQUIRES_OK(
      ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource));
  core::ScopedUnref unref(iterator_resource);
  OP_REQUIRES_OK(ctx, iterator_resource->SetIteratorFromDataset(ctx, dataset));
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
    background_worker_.Schedule([ctx, done]() {
      DatasetBase* dataset;
      OP_REQUIRES_OK_ASYNC(
          ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset), done);
      std::unique_ptr<IteratorBase> iterator;
      IteratorContext::Params params(ctx);
      std::unique_ptr<FunctionHandleCache> function_handle_cache =
          absl::make_unique<FunctionHandleCache>(params.lib);
      params.function_handle_cache = function_handle_cache.get();
      std::unique_ptr<ResourceMgr> resource_mgr =
          absl::make_unique<ResourceMgr>();
      params.resource_mgr = resource_mgr.get();
      IteratorContext iter_ctx(std::move(params));

      OP_REQUIRES_OK_ASYNC(
          ctx,
          dataset->MakeIterator(&iter_ctx, "SingleElementIterator", &iterator),
          done);

      // NOTE(jsimsa): We must destroy the iterator before calling `done()`, to
      // avoid destruction races.
      IteratorBase* raw_iterator = iterator.release();
      auto cleanup = gtl::MakeCleanup([ctx, raw_iterator, done] {
        delete raw_iterator;
        done();
      });
      std::vector<Tensor> components;
      components.reserve(dataset->output_dtypes().size());
      bool end_of_sequence = false;

      Status s =
          raw_iterator->GetNext(&iter_ctx, &components, &end_of_sequence);
      if (!s.ok()) {
        ctx->SetStatus(s);
        return;
      }
      if (end_of_sequence) {
        ctx->SetStatus(errors::InvalidArgument("Dataset was empty."));
        return;
      }
      for (int i = 0; i < components.size(); ++i) {
        // TODO(mrry): Check that the shapes match the shape attrs.
        ctx->set_output(i, components[i]);
      }

      components.clear();
      Status s2 =
          raw_iterator->GetNext(&iter_ctx, &components, &end_of_sequence);
      if (!s2.ok()) {
        ctx->SetStatus(s2);
        return;
      }
      if (!end_of_sequence) {
        ctx->SetStatus(
            errors::InvalidArgument("Dataset had more than one element."));
        return;
      }
    });
  }

 private:
  BackgroundWorker background_worker_;
};

class ReduceDatasetOp : public AsyncOpKernel {
 public:
  explicit ReduceDatasetOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(), "tf_data_reduce_dataset") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &reduce_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_inter_op_parallelism",
                                     &use_inter_op_parallelism_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    background_worker_.Schedule([this, ctx, done]() {
      DatasetBase* dataset;
      OP_REQUIRES_OK_ASYNC(
          ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset), done);
      OpInputList inputs;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("initial_state", &inputs),
                           done);
      std::vector<Tensor> state(inputs.begin(), inputs.end());

      std::unique_ptr<CapturedFunction> captured_func;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          CapturedFunction::Create(reduce_func_, ctx, "other_arguments",
                                   use_inter_op_parallelism_, &captured_func),
          done);

      IteratorContext::Params params(ctx);
      std::unique_ptr<FunctionHandleCache> function_handle_cache =
          absl::make_unique<FunctionHandleCache>(params.lib);
      params.function_handle_cache = function_handle_cache.get();
      std::unique_ptr<ResourceMgr> resource_mgr =
          absl::make_unique<ResourceMgr>();
      params.resource_mgr = resource_mgr.get();
      IteratorContext iter_ctx(std::move(params));
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          captured_func->Instantiate(&iter_ctx, &instantiated_captured_func),
          done);

      std::unique_ptr<IteratorBase> iterator;
      OP_REQUIRES_OK_ASYNC(
          ctx, dataset->MakeIterator(&iter_ctx, "ReduceIterator", &iterator),
          done);

      // NOTE(jsimsa): We must destroy the iterator before calling `done()`, to
      // avoid destruction races.
      IteratorBase* raw_iterator = iterator.release();
      auto cleanup = gtl::MakeCleanup([raw_iterator, done] {
        delete raw_iterator;
        done();
      });

      // Iterate through the input dataset.
      Status status;
      while (true) {
        std::vector<Tensor> next_input_element;
        bool end_of_input;
        status = raw_iterator->GetNext(&iter_ctx, &next_input_element,
                                       &end_of_input);
        if (!status.ok() || end_of_input) {
          break;
        }

        // Run the reduce function to update the current state.
        std::vector<Tensor> args;
        args.reserve(state.size() + next_input_element.size());
        std::copy(state.begin(), state.end(), std::back_inserter(args));
        std::copy(next_input_element.begin(), next_input_element.end(),
                  std::back_inserter(args));

        std::vector<Tensor> reduce_func_output;
        status = instantiated_captured_func->Run(&iter_ctx, std::move(args),
                                                 &reduce_func_output);
        if (!status.ok()) {
          break;
        }
        std::swap(reduce_func_output, state);
      }

      if (!status.ok()) {
        ctx->SetStatus(status);
        return;
      }
      for (int i = 0; i < state.size(); ++i) {
        OP_REQUIRES_ASYNC(
            ctx, state[i].dtype() == output_types_[i],
            errors::InvalidArgument(
                "The result does not match the expected type for component ", i,
                ". Expected: ", DataTypeString(output_types_[i]),
                ". Actual: ", DataTypeString(state[i].dtype()), "."),
            done);
        OP_REQUIRES_ASYNC(
            ctx, output_shapes_[i].IsCompatibleWith(state[i].shape()),
            errors::InvalidArgument(
                "The result does not match the expected shape for component ",
                i, ". Expected: ", output_shapes_[i].DebugString(),
                ". Actual: ", state[i].shape().DebugString(), "."),
            done);
        ctx->set_output(i, state[i]);
      }
    });
  }

 private:
  NameAttrList reduce_func_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool use_inter_op_parallelism_;
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
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

    FunctionLibraryRuntime* lib;
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    TF_RETURN_IF_ERROR(ctx->function_library()->Clone(&flib_def, &pflr, &lib));

    // Create an IteratorResource that will hold the iterator for this op.
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()->LookupOrCreate<IteratorResource>(
            cinfo->container(), cinfo->name(), iterator,
            [lib, this, &flib_def, &pflr](IteratorResource** ret)
                EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                  *ret = new IteratorResource(
                      output_dtypes_, output_shapes_, graph_def_version_,
                      nullptr, std::move(flib_def), std::move(pflr), lib);
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
    // Choose a step ID that is guaranteed not to clash with any
    // Session-generated step ID. DirectSession only generates
    // non-negative step IDs (contiguous, starting from 0), and
    // MasterSession generates 56-bit random step IDs whose MSB is
    // always 0, so a negative random step ID should suffice.
    opts.step_id = -std::abs(static_cast<int64>(random::New64()));
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

        Status s = iterator->GetNext(IteratorContext(ctx), &components,
                                     &end_of_sequence);
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

  OP_REQUIRES_OK(ctx, iterator->GetNext(IteratorContext(ctx), &components,
                                        &end_of_sequence));
  OP_REQUIRES(ctx, !end_of_sequence, errors::OutOfRange("End of sequence"));

  for (int i = 0; i < components.size(); ++i) {
    // TODO(mrry): Check that the shapes match the shape attrs.
    ctx->set_output(i, components[i]);
  }
}

namespace {

class IteratorGetNextAsOptionalOp : public AsyncOpKernel {
 public:
  explicit IteratorGetNextAsOptionalOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(),
                           "tf_data_iterator_get_next_as_optional") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
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

          Status s = iterator->GetNext(IteratorContext(ctx), &components,
                                       &end_of_sequence);
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
                  ctx,
                  output_shapes_[i].IsCompatibleWith(components[i].shape()),
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

 private:
  BackgroundWorker background_worker_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace

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
  string_handle_t->scalar<string>()() =
      resource_handle_t.scalar<ResourceHandle>()().SerializeAsString();
}

IteratorFromStringHandleOp::IteratorFromStringHandleOp(
    OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
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
      ctx, resource_handle.ParseFromString(string_handle_t.scalar<string>()()),
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
    Tensor* variant_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &variant_t));
    IteratorStateVariant v;
    OP_REQUIRES_OK(ctx, v.InitializeFromIterator(ctx, iterator_resource));
    variant_t->scalar<Variant>()() = v;
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
    Variant variant = ctx->input(1).scalar<Variant>()();
    auto* wrapper = variant.get<IteratorStateVariant>();
    OP_REQUIRES(ctx, wrapper != nullptr,
                errors::InvalidArgument(
                    "DeserializeIteratorOp: Unable to parse variant tensor."));
    OP_REQUIRES_OK(ctx, wrapper->status());
    OP_REQUIRES_OK(ctx, iterator_resource->Restore(ctx, wrapper->get()));
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
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIterator").Device(DEVICE_CPU).Priority(2),
    AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(
    Name("AnonymousIterator").Device(DEVICE_GPU).Priority(1),
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

}  // namespace

}  // namespace data
}  // namespace tensorflow
