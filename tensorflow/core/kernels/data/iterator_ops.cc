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
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/iterator.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following ops.

const char kIteratorVariantTypeName[] = "tensorflow::Iterator";

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != received[i]) {
      return errors::InvalidArgument("Data type mismatch at component ", i,
                                     ": expected ", DataTypeString(expected[i]),
                                     " but got ", DataTypeString(received[i]),
                                     ".");
    }
  }
  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!expected[i].IsCompatibleWith(received[i])) {
      return errors::InvalidArgument("Incompatible shapes at component ", i,
                                     ": expected ", expected[i].DebugString(),
                                     " but got ", received[i].DebugString(),
                                     ".");
    }
  }

  return Status::OK();
}

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
        flib_def_(std::move(flib_def)),
        pflr_(std::move(pflr)),
        lib_(lib),
        iterator_(nullptr),
        output_dtypes_(output_dtypes),
        output_shapes_(output_shapes) {}

  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) {
    std::shared_ptr<IteratorBase> captured_iterator(iterator_);
    if (captured_iterator) {
      if (lib_ != nullptr) {
        ctx->set_lib(lib_);
      }
      return captured_iterator->GetNext(ctx, out_tensors, end_of_sequence);
    } else {
      return errors::FailedPrecondition(
          "GetNext() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before getting the next element.");
    }
  }

  Status Save(OpKernelContext* ctx, IteratorStateWriter* writer) {
    std::shared_ptr<IteratorBase> captured_iterator(iterator_);
    if (captured_iterator) {
      return captured_iterator->Save(ctx, writer);
    } else {
      return errors::FailedPrecondition(
          "Save() failed because the iterator has not been initialized. "
          "Ensure that you have run the initializer operation for this "
          "iterator before saving it.");
    }
  }

  Status Restore(OpKernelContext* ctx, IteratorStateReader* reader) {
    string serialized_graph_def;
    TF_RETURN_IF_ERROR(reader->ReadScalar(GraphDatasetBase::kDatasetGraphKey,
                                          &serialized_graph_def));
    GraphDef graph_def;
    if (!graph_def.ParseFromString(serialized_graph_def)) {
      return errors::Internal("Error parsing dataset GraphDef.");
    }
    string output_node;
    TF_RETURN_IF_ERROR(reader->ReadScalar(
        GraphDatasetBase::kDatasetGraphOutputNodeKey, &output_node));
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
    std::unique_ptr<DeviceMgr> device_mgr(nullptr);
    std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
    std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
    TF_RETURN_IF_ERROR(ctx->function_library()->Clone(&flib_def, &pflr, &lib));
    TF_RETURN_IF_ERROR(flib_def->AddLibrary(graph_def.library()));

    TF_RETURN_IF_ERROR(
        graph_runner.Run(&graph, lib, {}, {output_node}, &outputs));
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(outputs[0], &dataset));

    IteratorContext iter_ctx = dataset::MakeIteratorContext(ctx);
    std::unique_ptr<IteratorBase> iterator;
    TF_RETURN_IF_ERROR(dataset->MakeIterator(&iter_ctx, "Iterator", &iterator));
    TF_RETURN_IF_ERROR(set_iterator(std::move(iterator)));
    std::shared_ptr<IteratorBase> captured_iterator(iterator_);

    if (captured_iterator) {
      IteratorContext::Params params;
      params.env = ctx->env();
      params.runner = *(ctx->runner());
      params.lib = lib;
      DeviceBase* device = lib->device();
      params.allocator_getter = [device](AllocatorAttributes attrs) {
        return device->GetAllocator(attrs);
      };
      IteratorContext iter_ctx(std::move(params));

      TF_RETURN_IF_ERROR(captured_iterator->Restore(&iter_ctx, reader));
      mutex_lock l(mu_);
      device_mgr_ = std::move(device_mgr);
      lib_def_ = std::move(flib_def);
      pflr_ = std::move(pflr);
      lib_ = lib;
      return Status::OK();
    } else {
      return errors::FailedPrecondition(
          "Failed to restore iterator. Make sure the checkpoint ",
          "is not corrupt. If the checkpoint does not contain the GraphDef, ",
          "you will need to initialize your iterator before restoring.");
    }
  }

  std::shared_ptr<const FunctionLibraryDefinition> function_library() {
    tf_shared_lock l(mu_);
    return lib_def_;
  }

  // Transfers ownership of iterator to this. This method is thread-safe.
  Status set_iterator(std::unique_ptr<IteratorBase> iterator) {
    if (iterator) {
      TF_RETURN_IF_ERROR(
          VerifyTypesMatch(output_dtypes_, iterator->output_dtypes()));
      TF_RETURN_IF_ERROR(
          VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));
    }
    iterator_.reset(iterator.release());
    return Status::OK();
  }


  std::shared_ptr<StatsAggregator> stats_aggregator() {
    tf_shared_lock l(mu_);
    return stats_aggregator_;
  }

  string DebugString() override { return "Iterator resource"; }

  const DataTypeVector& output_dtypes() const { return output_dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

 private:
  // The following (device_mgr_, flib_def_, pflr_) are only used when the
  // IteratorResource is shared between sessions and in that case we create
  // a new FLR. Otherwise these are set to null.
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* lib_ = nullptr;  // not owned.
  std::shared_ptr<IteratorBase> iterator_;
  mutex mu_;
  std::shared_ptr<StatsAggregator> stats_aggregator_ GUARDED_BY(mu_);
  std::shared_ptr<const FunctionLibraryDefinition> lib_def_ GUARDED_BY(mu_);
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
};

// Helper class for reading data from a VariantTensorData object.
class VariantTensorDataReader : public IteratorStateReader {
 public:
  explicit VariantTensorDataReader(const VariantTensorData* data)
      : data_(data) {
    PreProcess();
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
    return map_.find(key.ToString()) != map_.end();
  }

 private:
  void PreProcess() {
    string metadata;
    data_->get_metadata(&metadata);
    IteratorStateMetadata proto;
    if (!proto.ParseFromString(metadata)) {
      status_ = errors::Internal("Error parsing IteratorStateMetadata.");
      return;
    }
    size_t num_entries = proto.keys_size();
    CHECK_EQ(num_entries, data_->tensors_size());
    for (size_t i = 0; i < num_entries; i++) {
      map_[proto.keys(i)] = i;
    }
  }

  template <typename T>
  Status ReadScalarInternal(StringPiece key, T* val) {
    if (map_.find(key.ToString()) == map_.end()) {
      return errors::NotFound(key);
    }
    *val = data_->tensors(map_[key.ToString()]).scalar<T>()();
    return Status::OK();
  }

  Status ReadTensorInternal(StringPiece key, Tensor* val) {
    if (map_.find(key.ToString()) == map_.end()) {
      return errors::NotFound(key);
    }
    *val = data_->tensors(map_[key.ToString()]);
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

  // Writes the metadata to `data_`.
  Status Flush() {
    string metadata;
    if (!metadata_proto_.SerializeToString(&metadata)) {
      return errors::Internal("Unable to serialize IteratorStateMetadata.");
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
    // Write key to the metadata proto. This gets written to `data_`
    // when `Flush()` is called. We do this lazily to avoid multiple
    // serialization calls.
    metadata_proto_.add_keys(key.ToString());

    // Update tensors.
    *(data_->add_tensors()) = val;
    return Status::OK();
  }

  VariantTensorData* data_;
  // TODO(srbs): Set the version string.
  IteratorStateMetadata metadata_proto_;
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
    data_.reset(new VariantTensorData());
    data_->set_type_name(TypeName());
    VariantTensorDataWriter writer(data_.get());
    TF_RETURN_IF_ERROR(iterator_resource->Save(ctx, &writer));
    TF_RETURN_IF_ERROR(writer.Flush());
    return Status::OK();
  }
  string TypeName() const { return kIteratorVariantTypeName; }
  void Encode(VariantTensorData* data) const { *data = *data_; }
  bool Decode(const VariantTensorData& data) {
    if (data.type_name() != TypeName()) {
      return false;
    }
    std::unique_ptr<VariantTensorData> tensor_data(new VariantTensorData);
    *tensor_data = data;
    std::unique_ptr<VariantTensorDataReader> reader(
        new VariantTensorDataReader(tensor_data.get()));
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

// Note that IteratorHandleOp holds a reference to the resource it creates. If
// cleaning up resources with DestroyResourceOp is important, consider creating
// resource containers with AnonymousIteratorHandleOp instead.
class IteratorHandleOp : public OpKernel {
 public:
  explicit IteratorHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel. Ideally the resource should be deleted when it is no longer held
  // by anyone, but it would break backward compatibility.
  ~IteratorHandleOp() override {
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

  void Compute(OpKernelContext* context) override LOCKS_EXCLUDED(mu_) {
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
                [lib, &device_mgr, &flib_def, &pflr,
                 this](IteratorResource** ret) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
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

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  Status VerifyResource(IteratorResource* resource) {
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_dtypes_, resource->output_dtypes()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
    return Status::OK();
  }

  template <typename To, typename From>  // use like this: down_cast<T*>(foo);
  static inline To down_cast(From* f) {  // so we only accept pointers
    static_assert(
        (std::is_base_of<From, typename std::remove_pointer<To>::type>::value),
        "target type not derived from source type");

    // We skip the assert and hence the dynamic_cast if RTTI is disabled.
#if !defined(__GNUC__) || defined(__GXX_RTTI)
    // Uses RTTI in dbg and fastbuild. asserts are disabled in opt builds.
    assert(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif  // !defined(__GNUC__) || defined(__GXX_RTTI)
    return static_cast<To>(f);
  }

  FunctionLibraryRuntime* CreatePrivateFLR(
      OpKernelContext* ctx, std::unique_ptr<DeviceMgr>* device_mgr,
      std::unique_ptr<FunctionLibraryDefinition>* flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime>* pflr) {
    // Wrap the existing device in order to see any captured resources
    // in its resource manager. The existing device will outlive the
    // IteratorResource, because we are storing the IteratorResource
    // in that device's resource manager.
    Device* wrapped_device = RenamedDevice::NewRenamedDevice(
        ctx->device()->name(), down_cast<Device*>(ctx->device()),
        false /* owns_underlying */, false /* isolate_session_state */);
    device_mgr->reset(new DeviceMgr({wrapped_device}));
    flib_def->reset(new FunctionLibraryDefinition(
        *ctx->function_library()->GetFunctionLibraryDefinition()));
    pflr->reset(new ProcessFunctionLibraryRuntime(
        device_mgr->get(), ctx->env(), graph_def_version_, flib_def->get(),
        {} /* TODO(mrry): OptimizerOptions? */,
        nullptr /* TODO(mrry): ClusterFLR */));

    return (*pflr)->GetFLR(ctx->device()->name());
  }

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  IteratorResource* resource_ GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
  string name_;
};

// Like IteratorHandleOp, but creates handles which are never shared, and does
// not hold a reference to these handles. The latter is important for eager
// execution, since OpKernel instances generally live as long as the program
// running them.
class AnonymousIteratorHandleOp : public OpKernel {
 public:
  explicit AnonymousIteratorHandleOp(OpKernelConstruction* context)
      : OpKernel(context), graph_def_version_(context->graph_def_version()) {
    OP_REQUIRES_OK(context, context->GetAttr("output_types", &output_dtypes_));
    OP_REQUIRES_OK(context, context->GetAttr("output_shapes", &output_shapes_));
  }

  void Compute(OpKernelContext* context) override {
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
        Status status = mgr->Lookup<IteratorResource>(
            container_name, unique_name, &existing_resource);
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

 private:
  // Coordinates Iterator unique name creation across AnonymousIteratorHandleOp
  // instances.
  static mutex static_resource_lookup_mutex_;
  // current_id_ is just a hint for creating unique names. If it turns out
  // there's a collision (e.g. because another AnonymousIteratorHandleOp
  // instance is generating handles) we'll just skip that id.
  static int64 current_id_ GUARDED_BY(static_resource_lookup_mutex_);
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
};

// Static initializers for AnonymousIteratorHandleOp id counting.
mutex AnonymousIteratorHandleOp::static_resource_lookup_mutex_{
    LINKER_INITIALIZED};
int64 AnonymousIteratorHandleOp::current_id_(0);

class MakeIteratorOp : public OpKernel {
 public:
  explicit MakeIteratorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
    IteratorResource* iterator_resource;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 1), &iterator_resource));
    core::ScopedUnref unref(iterator_resource);

    IteratorContext iter_ctx = dataset::MakeIteratorContext(ctx);
    std::unique_ptr<IteratorBase> iterator;
    OP_REQUIRES_OK(ctx,
                   dataset->MakeIterator(&iter_ctx, "Iterator", &iterator));
    OP_REQUIRES_OK(ctx, iterator_resource->set_iterator(std::move(iterator)));
  }
};

class ToSingleElementOp : public AsyncOpKernel {
 public:
  explicit ToSingleElementOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("to_single_element_op_thread_",
                            SanitizeThreadSuffix(name())),
            1 /* num_threads */, false /* low_latency_hint */)) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    thread_pool_->Schedule([ctx, done]() {
      DatasetBase* dataset;
      OP_REQUIRES_OK_ASYNC(
          ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset), done);
      IteratorContext iter_ctx = dataset::MakeIteratorContext(ctx);
      std::unique_ptr<IteratorBase> iterator;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          dataset->MakeIterator(&iter_ctx, "SingleElementIterator", &iterator),
          done);
      std::vector<Tensor> components;
      components.reserve(dataset->output_dtypes().size());
      bool end_of_sequence;

      OP_REQUIRES_OK_ASYNC(
          ctx, iterator->GetNext(&iter_ctx, &components, &end_of_sequence),
          done);
      OP_REQUIRES_ASYNC(ctx, !end_of_sequence,
                        errors::InvalidArgument("Dataset was empty."), done);

      for (int i = 0; i < components.size(); ++i) {
        // TODO(mrry): Check that the shapes match the shape attrs.
        ctx->set_output(i, components[i]);
      }

      components.clear();
      OP_REQUIRES_OK_ASYNC(
          ctx, iterator->GetNext(&iter_ctx, &components, &end_of_sequence),
          done);
      OP_REQUIRES_ASYNC(
          ctx, end_of_sequence,
          errors::InvalidArgument("Dataset had more than one element."), done);

      done();
    });
  }

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class OneShotIteratorOp : public AsyncOpKernel {
 public:
  explicit OneShotIteratorOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("one_shot_iterator_initialization_thread_",
                            SanitizeThreadSuffix(name())),
            1 /* num_threads */, false /* low_latency_hint */)),
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
          thread_pool_->Schedule([this, ctx, done]() { Init(ctx, done); });
          initialization_started_ = true;
        } else {
          done_callbacks_.emplace_back(ctx, std::move(done));
        }
        return;
      }
    }
    ProduceOutput(ctx, std::move(done));
  }

 private:
  void Init(OpKernelContext* ctx, DoneCallback done) {
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
      ProduceOutput(ctx_done.first, std::move(ctx_done.second));
    }
    ProduceOutput(ctx, std::move(done));
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
    IteratorContext iter_ctx = dataset::MakeIteratorContext(ctx);
    std::unique_ptr<IteratorBase> iter;
    TF_RETURN_IF_ERROR(dataset->MakeIterator(&iter_ctx, "Iterator", &iter));
    TF_RETURN_IF_ERROR((*iterator)->set_iterator(std::move(iter)));

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

  std::unique_ptr<thread::ThreadPool> thread_pool_;

  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  IteratorResource* iterator_resource_ GUARDED_BY(mu_) = nullptr;

  bool initialization_started_ GUARDED_BY(mu_) = false;
  Status initialization_status_ GUARDED_BY(mu_);
  std::vector<std::pair<OpKernelContext*, DoneCallback>> done_callbacks_
      GUARDED_BY(mu_);
  const int graph_def_version_;
};

class IteratorGetNextOp : public AsyncOpKernel {
 public:
  explicit IteratorGetNextOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("iterator_get_next_thread_",
                            SanitizeThreadSuffix(name())),
            1 /* num_threads */, false /* low_latency_hint */)) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    IteratorResource* iterator;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator), done);
    // The call to `iterator->GetNext()` may block and depend on an
    // inter-op thread pool thread, so we issue the call from the
    // owned thread pool.
    thread_pool_->Schedule(std::bind(
        [ctx, iterator](DoneCallback done) {
          std::vector<Tensor> components;
          bool end_of_sequence = false;

          IteratorContext::Params params;
          params.env = ctx->env();
          params.stats_aggregator_getter = [iterator]() {
            return iterator->stats_aggregator();
          };
          params.runner = *(ctx->runner());
          params.function_library = iterator->function_library();
          DeviceBase* device = ctx->function_library()->device();
          params.allocator_getter = [device](AllocatorAttributes attrs) {
            return device->GetAllocator(attrs);
          };
          IteratorContext iter_ctx(std::move(params));

          Status s =
              iterator->GetNext(&iter_ctx, &components, &end_of_sequence);
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

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class IteratorGetNextSyncOp : public OpKernel {
 public:
  explicit IteratorGetNextSyncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    IteratorResource* iterator;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &iterator));
    core::ScopedUnref unref_iterator(iterator);

    std::vector<Tensor> components;
    bool end_of_sequence = false;

    IteratorContext::Params params;
    params.env = ctx->env();
    params.stats_aggregator_getter = [iterator]() {
      return iterator->stats_aggregator();
    };
    params.runner = *(ctx->runner());
    params.function_library = iterator->function_library();
    DeviceBase* device = ctx->function_library()->device();
    params.allocator_getter = [device](AllocatorAttributes attrs) {
      return device->GetAllocator(attrs);
    };
    IteratorContext iter_ctx(std::move(params));

    OP_REQUIRES_OK(ctx,
                   iterator->GetNext(&iter_ctx, &components, &end_of_sequence));
    OP_REQUIRES(ctx, !end_of_sequence, errors::OutOfRange("End of sequence"));

    for (int i = 0; i < components.size(); ++i) {
      // TODO(mrry): Check that the shapes match the shape attrs.
      ctx->set_output(i, components[i]);
    }
  }
};

class IteratorToStringHandleOp : public OpKernel {
 public:
  explicit IteratorToStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
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
};

class IteratorFromStringHandleOp : public OpKernel {
 public:
  explicit IteratorFromStringHandleOp(OpKernelConstruction* ctx)
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

  void Compute(OpKernelContext* ctx) override {
    const Tensor& string_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(string_handle_t.shape()),
                errors::InvalidArgument("string_handle must be a scalar"));

    ResourceHandle resource_handle;
    OP_REQUIRES(
        ctx,
        resource_handle.ParseFromString(string_handle_t.scalar<string>()()),
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
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, resource_handle, &iterator_resource));
    core::ScopedUnref unref_iterator(iterator_resource);
    if (!output_dtypes_.empty()) {
      OP_REQUIRES_OK(ctx, VerifyTypesMatch(output_dtypes_,
                                           iterator_resource->output_dtypes()));
    }
    if (!output_shapes_.empty()) {
      OP_REQUIRES_OK(
          ctx, VerifyShapesCompatible(output_shapes_,
                                      iterator_resource->output_shapes()));
    }

    Tensor* resource_handle_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &resource_handle_t));
    resource_handle_t->scalar<ResourceHandle>()() = resource_handle;
  }

 private:
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

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
REGISTER_KERNEL_BUILDER(Name("MakeIterator").Device(DEVICE_CPU),
                        MakeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("AnonymousIterator").Device(DEVICE_CPU),
                        AnonymousIteratorHandleOp);
REGISTER_KERNEL_BUILDER(Name("DatasetToSingleElement").Device(DEVICE_CPU),
                        ToSingleElementOp);
REGISTER_KERNEL_BUILDER(Name("OneShotIterator").Device(DEVICE_CPU),
                        OneShotIteratorOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNext").Device(DEVICE_CPU),
                        IteratorGetNextOp);
REGISTER_KERNEL_BUILDER(Name("IteratorGetNextSync").Device(DEVICE_CPU),
                        IteratorGetNextSyncOp);
REGISTER_KERNEL_BUILDER(Name("IteratorToStringHandle").Device(DEVICE_CPU),
                        IteratorToStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("IteratorFromStringHandle").Device(DEVICE_CPU),
                        IteratorFromStringHandleOp);
REGISTER_KERNEL_BUILDER(Name("SerializeIterator").Device(DEVICE_CPU),
                        SerializeIteratorOp);
REGISTER_KERNEL_BUILDER(Name("DeserializeIterator").Device(DEVICE_CPU),
                        DeserializeIteratorOp);

}  // namespace

}  // namespace tensorflow
