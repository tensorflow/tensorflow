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
#include "tensorflow/core/framework/dataset.h"

#include <unordered_map>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/resource.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
namespace data {
namespace {

static mutex* get_dataset_op_registry_lock() {
  static mutex dataset_op_registry_lock(LINKER_INITIALIZED);
  return &dataset_op_registry_lock;
}

static std::unordered_set<string>* get_dataset_op_registry() {
  static std::unordered_set<string>* names = new std::unordered_set<string>;
  return names;
}

// A wrapper class for storing a `DatasetBase` instance in a DT_VARIANT tensor.
// Objects of the wrapper class own a reference on an instance of `DatasetBase`,
// and the wrapper's copy constructor and destructor take care of managing the
// reference count.
//
// NOTE(mrry): This is not a feature-complete implementation of the DT_VARIANT
// specification. In particular, we cannot currently serialize an arbitrary
// `DatasetBase` object, so the `Encode()` and `Decode()` methods are not
// implemented.
class DatasetVariantWrapper {
 public:
  DatasetVariantWrapper() : dataset_(nullptr) {}

  // Transfers ownership of `dataset` to `*this`.
  explicit DatasetVariantWrapper(DatasetBase* dataset) : dataset_(dataset) {}

  DatasetVariantWrapper(const DatasetVariantWrapper& other)
      : dataset_(other.dataset_) {
    if (dataset_) dataset_->Ref();
  }

  DatasetVariantWrapper& operator=(DatasetVariantWrapper&& other) {
    if (&other == this) return *this;
    std::swap(dataset_, other.dataset_);
    return *this;
  }

  DatasetVariantWrapper& operator=(const DatasetVariantWrapper& other) = delete;

  ~DatasetVariantWrapper() {
    if (dataset_) dataset_->Unref();
  }

  DatasetBase* get() const { return dataset_; }

  string TypeName() const { return "tensorflow::DatasetVariantWrapper"; }
  string DebugString() const {
    if (dataset_) {
      return dataset_->DebugString();
    } else {
      return "<Uninitialized DatasetVariantWrapper>";
    }
  }
  void Encode(VariantTensorData* data) const {
    LOG(ERROR) << "The Encode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
  }
  bool Decode(const VariantTensorData& data) {
    LOG(ERROR) << "The Decode() method is not implemented for "
                  "DatasetVariantWrapper objects.";
    return false;
  }

 private:
  DatasetBase* dataset_;  // Owns one reference.
};

const char kWrappedDatasetVariantTypeName[] =
    "tensorflow::data::WrappedDatasetVariant";

class WrappedDatasetVariantWrapper {
 public:
  WrappedDatasetVariantWrapper() {}

  explicit WrappedDatasetVariantWrapper(const Tensor& ds_tensor)
      : ds_tensor_(ds_tensor) {}

  Tensor get() const { return ds_tensor_; }

  string TypeName() const { return "tensorflow::WrappedDatasetVariantWrapper"; }

  string DebugString() const {
    return "tensorflow::WrappedDatasetVariantWrapper::DebugString";
  }

  void Encode(VariantTensorData* data) const {
    *(data->add_tensors()) = ds_tensor_;
  }

  bool Decode(const VariantTensorData& data) {
    ds_tensor_ = data.tensors(0);
    return true;
  }

 private:
  Tensor ds_tensor_;
};

class WrapDatasetVariantOp : public OpKernel {
 public:
  explicit WrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    DatasetBase* unused;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(tensor, &unused));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->scalar<Variant>()() = WrappedDatasetVariantWrapper(tensor);
  }
};

REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant").Device(DEVICE_CPU),
                        WrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("WrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        WrapDatasetVariantOp);

class UnwrapDatasetVariantOp : public OpKernel {
 public:
  explicit UnwrapDatasetVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                tensor.dtype() == DT_VARIANT &&
                    TensorShapeUtils::IsScalar(tensor.shape()),
                errors::InvalidArgument(
                    "Dataset tensor must be a scalar of dtype DT_VARIANT."));
    Variant variant = tensor.scalar<Variant>()();
    const WrappedDatasetVariantWrapper* wrapper =
        variant.get<WrappedDatasetVariantWrapper>();
    OP_REQUIRES(ctx, wrapper != nullptr,
                errors::InvalidArgument(
                    "Tensor must be a WrappedDataset variant object."));
    Tensor ds_tensor = wrapper->get();
    OP_REQUIRES_OK(ctx, ctx->set_output("output_handle", ds_tensor));
  }
};

REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant").Device(DEVICE_CPU),
                        UnwrapDatasetVariantOp);
REGISTER_KERNEL_BUILDER(Name("UnwrapDatasetVariant")
                            .HostMemory("input_handle")
                            .HostMemory("output_handle")
                            .Device(DEVICE_GPU),
                        UnwrapDatasetVariantOp);

static Status WrappedDatasetVariantDeviceCopy(
    const WrappedDatasetVariantWrapper& from, WrappedDatasetVariantWrapper* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
  *to = WrappedDatasetVariantWrapper(from);
  return Status::OK();
}

#define REGISTER_OPTIONAL_COPY(DIRECTION)               \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      WrappedDatasetVariantWrapper, DIRECTION,          \
      WrappedDatasetVariantDeviceCopy)

REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_OPTIONAL_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(WrappedDatasetVariantWrapper,
                                       kWrappedDatasetVariantTypeName);

}  // namespace

Status GraphDefBuilderWrapper::AddDataset(
    const DatasetBase* dataset,
    const std::vector<std::pair<size_t, Node*>>& inputs,
    const std::vector<std::pair<size_t, gtl::ArraySlice<Node*>>>& list_inputs,
    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
    Node** output) {
  const string& type_string = dataset->type_string();
  std::unique_ptr<const GraphDefBuilder::Options> opts(
      new GraphDefBuilder::Options(b_->opts()));
  // TODO(srbs|mrry): Not all datasets have output_types and output_shapes
  // attributes defined. It will be nice to have a consistent pattern.
  bool has_output_types_attr = HasAttr(type_string, "output_types");
  bool has_output_shapes_attr = HasAttr(type_string, "output_shapes");
  if (has_output_shapes_attr) {
    opts.reset(new GraphDefBuilder::Options(
        opts->WithAttr("output_shapes", dataset->output_shapes())));
  }
  if (has_output_types_attr) {
    opts.reset(new GraphDefBuilder::Options(
        opts->WithAttr("output_types", dataset->output_dtypes())));
  }
  for (const auto& attr : attrs) {
    opts.reset(
        new GraphDefBuilder::Options(opts->WithAttr(attr.first, attr.second)));
  }
  if (opts->HaveError()) {
    return errors::Internal("AddDataset: Failed to build Options with error ",
                            opts->StatusToString());
  }
  NodeBuilder node_builder(opts->GetNameForOp(type_string), type_string,
                           opts->op_registry());
  {
    size_t total_size = inputs.size() + list_inputs.size();
    auto inputs_iter = inputs.begin();
    auto list_inputs_iter = list_inputs.begin();
    for (int i = 0; i < total_size; i++) {
      if (inputs_iter != inputs.end() && inputs_iter->first == i) {
        node_builder.Input(NodeBuilder::NodeOut(inputs_iter->second));
        inputs_iter++;
      } else if (list_inputs_iter != list_inputs.end() &&
                 list_inputs_iter->first == i) {
        std::vector<NodeBuilder::NodeOut> nodeout_inputs;
        nodeout_inputs.reserve(list_inputs_iter->second.size());
        for (Node* n : list_inputs_iter->second) {
          nodeout_inputs.emplace_back(n);
        }
        node_builder.Input(nodeout_inputs);
        list_inputs_iter++;
      } else {
        return errors::InvalidArgument("No input found for index ", i);
      }
    }
  }
  *output = opts->FinalizeBuilder(&node_builder);
  if (*output == nullptr) {
    return errors::Internal("AddDataset: Failed to build ", type_string,
                            " op with error ", opts->StatusToString());
  }
  return Status::OK();
}

Status GraphDefBuilderWrapper::AddFunction(
    SerializationContext* ctx, const string& function_name,
    const FunctionLibraryDefinition& lib_def) {
  if (b_->HasFunction(function_name)) {
    VLOG(1) << "Function with name " << function_name << "already exists in"
            << " the graph. It will not be added again.";
    return Status::OK();
  }
  const FunctionDef* f_def = lib_def.Find(function_name);
  if (f_def == nullptr) {
    return errors::InvalidArgument("Unable to find FunctionDef for ",
                                   function_name, " in the registry.");
  }
  FunctionDefLibrary def;
  *def.add_function() = *f_def;
  const string gradient_func = lib_def.FindGradient(function_name);
  if (!gradient_func.empty()) {
    GradientDef* g_def = def.add_gradient();
    g_def->set_function_name(function_name);
    g_def->set_gradient_func(gradient_func);
  }
  TF_RETURN_IF_ERROR(b_->AddFunctionLibrary(def));

  // Recursively add functions in inputs of function_name.
  for (const NodeDef& node_def : f_def->node_def()) {
    const OpRegistrationData* op_reg_data = nullptr;
    TF_RETURN_IF_ERROR(lib_def.LookUp(node_def.op(), &op_reg_data));
    if (op_reg_data->is_function_op) {
      TF_RETURN_IF_ERROR(AddFunction(ctx, op_reg_data->op_def.name(), lib_def));
    }
    // Recursively add functions in attrs of this NodeDef.
    for (const auto& pair : node_def.attr()) {
      TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, pair.second, lib_def));
    }
  }

  // Recursively add functions in attrs of function_name.
  for (auto iter = f_def->attr().begin(); iter != f_def->attr().end(); iter++) {
    TF_RETURN_IF_ERROR(AddAttrFunctions(ctx, iter->second, lib_def));
  }
  return Status::OK();
}

void GraphDefBuilderWrapper::AddPlaceholderInternal(const Tensor& val,
                                                    Node** output) {
  *output = ops::SourceOp(
      "Placeholder",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("shape", val.shape()));
}

void GraphDefBuilderWrapper::AddTensorInternal(const Tensor& val,
                                               Node** output) {
  *output = ops::SourceOp(
      "Const",
      b_->opts().WithAttr("dtype", val.dtype()).WithAttr("value", val));
}

bool GraphDefBuilderWrapper::HasAttr(const string& name,
                                     const string& attr_name) const {
  const OpDef* op_def = nullptr;
  Status s = b_->opts().op_registry()->LookUpOpDef(name, &op_def);
  if (!s.ok() || op_def == nullptr) {
    return false;
  }
  return HasAttr(op_def, attr_name);
}

Status IteratorBase::InitializeBase(IteratorContext* ctx,
                                    const IteratorBase* parent) {
  parent_ = parent;
  id_ =
      Hash64CombineUnordered(Hash64(prefix()), reinterpret_cast<uint64>(this));
  if (parent_) {
    parent_id_ = Hash64CombineUnordered(Hash64(parent_->prefix()),
                                        reinterpret_cast<uint64>(parent_));
  }
  if (const auto& model = ctx->model()) {
    auto factory = [ctx, this](model::Node::Args args) {
      return CreateNode(ctx, std::move(args));
    };
    model->AddNode(std::move(factory), prefix(), parent->model_node(), &node_);
    cleanup_fns_.push_back([this, model]() { model->RemoveNode(node_); });
  }
  return Status::OK();
}

int64 GetAllocatedBytes(const std::vector<Tensor>& element) {
  int64 allocated_bytes = 0;
  DatasetBase* dataset;
  for (auto& tensor : element) {
    if (tensor.dtype() == DT_VARIANT &&
        GetDatasetFromVariantTensor(tensor, &dataset).ok()) {
      allocated_bytes += dataset->AllocatedBytes();
    } else {
      allocated_bytes += tensor.AllocatedBytes();
    }
  }
  return allocated_bytes;
}

int64 GetTotalBytes(const std::vector<Tensor>& element) {
  int64 total_bytes = 0;
  DatasetBase* dataset;
  for (auto& tensor : element) {
    if (tensor.dtype() == DT_VARIANT &&
        GetDatasetFromVariantTensor(tensor, &dataset).ok()) {
      total_bytes += dataset->TotalBytes();
    } else {
      total_bytes += tensor.TotalBytes();
    }
  }
  return total_bytes;
}

Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset) {
  if (!(tensor.dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor.shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  const Variant& variant = tensor.scalar<Variant>()();
  const DatasetVariantWrapper* wrapper = variant.get<DatasetVariantWrapper>();
  if (wrapper == nullptr) {
    return errors::InvalidArgument("Tensor must be a Dataset object.");
  }
  *out_dataset = wrapper->get();
  if (*out_dataset == nullptr) {
    return errors::Internal("Read uninitialized Dataset variant.");
  }
  return Status::OK();
}

Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor) {
  if (!(tensor->dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(tensor->shape()))) {
    return errors::InvalidArgument(
        "Dataset tensor must be a scalar of dtype DT_VARIANT.");
  }
  tensor->scalar<Variant>()() = DatasetVariantWrapper(dataset);
  return Status::OK();
}

Status DatasetBase::MakeIterator(
    IteratorContext* ctx, const IteratorBase* parent,
    const string& output_prefix,
    std::unique_ptr<IteratorBase>* iterator) const {
  *iterator = MakeIteratorInternal(output_prefix);
  Status s = (*iterator)->InitializeBase(ctx, parent);
  if (s.ok()) {
    s.Update((*iterator)->Initialize(ctx));
  }
  if (!s.ok()) {
    // Reset the iterator to avoid returning an uninitialized iterator.
    iterator->reset();
  }
  return s;
}

Status DatasetBase::DatasetGraphDefBuilder::AddInputDataset(
    SerializationContext* ctx, const DatasetBase* dataset, Node** output) {
  Status status = dataset->AsGraphDefInternal(ctx, this, output);
  if (errors::IsUnimplemented(status) && !ctx->fail_if_unimplemented()) {
    Tensor t(DT_VARIANT, TensorShape({}));
    // `StoreDatasetInVariantTensor` will transfer ownership of `dataset`. We
    // increment the refcount of `dataset` here to retain ownership.
    dataset->Ref();
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(const_cast<DatasetBase*>(dataset), &t));
    TF_RETURN_IF_ERROR(AddPlaceholder(t, output));
    DCHECK_NE(ctx->input_list(), nullptr);
    ctx->input_list()->emplace_back((*output)->name(), std::move(t));
    LOG_EVERY_N_SEC(WARNING, 30)
        << "Input of " << dataset->DebugString()
        << " will not be optimized because the dataset does not implement the "
           "AsGraphDefInternal() method needed to apply optimizations.";
    return Status::OK();
  }
  return status;
}

DatasetBaseIterator::DatasetBaseIterator(const BaseParams& params)
    : params_(params) {
  params_.dataset->Ref();
  VLOG(2) << prefix() << " constructor";
}

DatasetBaseIterator::~DatasetBaseIterator() {
  VLOG(2) << prefix() << " destructor";
  params_.dataset->Unref();
}

string DatasetBaseIterator::BuildTraceMeName() {
  string result = strings::StrCat(params_.prefix, "#id=", id_);
  if (parent_) {
    strings::StrAppend(&result, ",parent_id=", parent_id_);
  }

  TraceMeMetadata metadata = GetTraceMeMetadata();
  for (const auto& pair : metadata) {
    strings::StrAppend(&result, ",", pair.first, "=", pair.second);
  }
  strings::StrAppend(&result, "#");
  return result;
}

Status DatasetBaseIterator::GetNext(IteratorContext* ctx,
                                    std::vector<Tensor>* out_tensors,
                                    bool* end_of_sequence) {
  profiler::TraceMe activity([&] { return BuildTraceMeName(); },
                             profiler::TraceMeLevel::kInfo);
  DVLOG(3) << prefix() << " GetNext enter";
  RecordInput(ctx);
  RecordStart(ctx, /*stop_output=*/true);
  Status s = GetNextInternal(ctx, out_tensors, end_of_sequence);
  if (s.ok() && !*end_of_sequence) RecordElement(ctx, out_tensors);
  RecordStop(ctx, /*start_output=*/true);
  if (TF_PREDICT_FALSE(errors::IsOutOfRange(s))) {
    s = errors::Internal("Iterator \"", params_.prefix,
                         "\" returned `OutOfRange`. This indicates an "
                         "implementation error as `OutOfRange` errors are not "
                         "expected to be returned here. Original message: ",
                         s.error_message());
    LOG(ERROR) << s;
  }
  DVLOG(3) << prefix() << " GetNext exit";
  return s;
}

void DatasetOpKernel::Compute(OpKernelContext* ctx) {
  DatasetBase* dataset = nullptr;
  MakeDataset(ctx, &dataset);
  if (ctx->status().ok()) {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    OP_REQUIRES_OK(ctx, StoreDatasetInVariantTensor(dataset, output));
  }
}

string DatasetOpKernel::TraceString(OpKernelContext* ctx, bool verbose) {
  return strings::StrCat(name_view(), ":", type_string_view());
}

// static
bool DatasetOpKernel::IsDatasetOp(const OpDef* op_def) {
  if (DatasetOpRegistry::IsRegistered(op_def->name())) {
    return true;
  }

  return (op_def->output_arg_size() == 1 &&
          op_def->output_arg(0).type() == DT_VARIANT &&
          (absl::EndsWith(op_def->name(), "Dataset") ||
           absl::EndsWith(op_def->name(), "DatasetV2")));
}

void UnaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  MakeDataset(ctx, input, output);
}

void BinaryDatasetOpKernel::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase** output) {
  DatasetBase* input;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input));
  DatasetBase* another_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(1), &another_input));
  MakeDataset(ctx, input, another_input, output);
}

const char DatasetBase::kDatasetGraphKey[] = "_DATASET_GRAPH";
const char DatasetBase::kDatasetGraphOutputNodeKey[] =
    "_DATASET_GRAPH_OUTPUT_NODE";

BackgroundWorker::BackgroundWorker(Env* env, const char* name)
    : env_(env), name_(name) {}

BackgroundWorker::~BackgroundWorker() {
  {
    mutex_lock l(mu_);
    cancelled_ = true;
  }
  cond_var_.notify_one();
  // Block until the background thread has terminated.
  //
  // NOTE(mrry): We explicitly free and join the thread here because
  // `WorkerLoop()` uses other members of this object, and so we must join
  // the thread before destroying them.
  thread_.reset();
}

void BackgroundWorker::Schedule(std::function<void()> work_item) {
  {
    mutex_lock l(mu_);
    if (!thread_) {
      thread_ = absl::WrapUnique(env_->StartThread(
          {} /* thread_options */, name_, [this]() { WorkerLoop(); }));
    }
    work_queue_.push_back(std::move(work_item));
  }
  cond_var_.notify_one();
}

void BackgroundWorker::WorkerLoop() {
  tensorflow::ResourceTagger tag(kTFDataResourceTag, "Background");
  while (true) {
    std::function<void()> work_item = nullptr;
    {
      mutex_lock l(mu_);
      while (!cancelled_ && work_queue_.empty()) {
        cond_var_.wait(l);
      }
      if (cancelled_) {
        return;
      }
      DCHECK(!work_queue_.empty());
      work_item = std::move(work_queue_.front());
      work_queue_.pop_front();
    }
    DCHECK(work_item != nullptr);
    work_item();
  }
}

// static
void DatasetOpRegistry::Register(const string& op_name) {
  mutex_lock l(*get_dataset_op_registry_lock());
  get_dataset_op_registry()->insert(op_name);
}

// static
bool DatasetOpRegistry::IsRegistered(const string& op_name) {
  mutex_lock l(*get_dataset_op_registry_lock());
  std::unordered_set<string>* op_names = get_dataset_op_registry();
  return op_names->find(op_name) != op_names->end();
}

namespace {
class RunnerImpl : public Runner {
 public:
  void Run(const std::function<void()>& f) override {
    tensorflow::ResourceTagger tag(kTFDataResourceTag, "Runner");
    f();

    // NOTE: We invoke a virtual function to prevent `f` being tail-called, and
    // thus ensure that this function remains on the stack until after `f`
    // returns.
    PreventTailCall();
  }

 private:
  virtual void PreventTailCall() {}
};
}  // namespace

/* static */
Runner* Runner::get() {
  static Runner* singleton = new RunnerImpl;
  return singleton;
}

}  // namespace data
}  // namespace tensorflow
