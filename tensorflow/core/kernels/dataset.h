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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATASET_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATASET_H_

#include <memory>

#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

// Polymorphic datasets should support all primitive TensorFlow
// types. Use this macro to expand `m(T)` once for each primitive type
// `T`, e.g. to build a `switch` statement.
#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)

namespace tensorflow {

class ResourceMgr;

// Interface for reading values from a key-value store.
// Used for restoring iterator state.
class IteratorStateReader {
 public:
  virtual Status ReadScalar(StringPiece key, int64* val) = 0;
  virtual Status ReadScalar(StringPiece key, string* val) = 0;
  virtual Status ReadTensor(StringPiece key, Tensor* val) = 0;
  virtual bool Contains(StringPiece key) = 0;

  virtual ~IteratorStateReader() {}
};

// Interface for writing values to a key-value store.
// Used for saving iterator state.
class IteratorStateWriter {
 public:
  virtual Status WriteScalar(StringPiece key, const int64 val) = 0;
  virtual Status WriteScalar(StringPiece key, const string& val) = 0;
  virtual Status WriteTensor(StringPiece key, const Tensor& val) = 0;

  virtual ~IteratorStateWriter() {}
};

// Wrapper around GraphDefBuilder. Used to serialize Dataset graph.
class GraphDefBuilderWrapper {
 public:
  explicit GraphDefBuilderWrapper(GraphDefBuilder* b) : b_(b) {}

  // Adds a Const node with scalar value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  template <typename T>
  Status AddScalar(const T& val, Node** output) {
    Tensor val_t = Tensor(DataTypeToEnum<T>::v(), TensorShape({}));
    val_t.scalar<T>()() = val;
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddScalar: Failed to build Const op.");
    }
    return Status::OK();
  }

  // Adds a Const node with vector value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  template <typename T>
  Status AddVector(const std::vector<T>& val, Node** output) {
    Tensor val_t = Tensor(DataTypeToEnum<T>::v(),
                          TensorShape({static_cast<int64>(val.size())}));
    for (int i = 0; i < val.size(); i++) {
      val_t.flat<T>()(i) = val[i];
    }
    AddTensorInternal(val_t, output);
    if (*output == nullptr) {
      return errors::Internal("AddVector: Failed to build Const op.");
    }
    return Status::OK();
  }

  // Adds a Const node with Tensor value to the Graph.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  Status AddTensor(const Tensor& val, Node** output) {
    AddTensorInternal(val, output);
    if (*output == nullptr) {
      return errors::Internal("AddTesor: Failed to build Const op.");
    }
    return Status::OK();
  }

  template <class DatasetType>
  Status AddDataset(const DatasetType* dataset,
                    const std::vector<NodeBuilder::NodeOut>& inputs,
                    Node** output) {
    return AddDataset(dataset, inputs, {}, output);
  }

  // Adds a node corresponding to the `DatasetType` to the Graph.
  // Return value of `DatasetType::op_name()` is used as the op type for the
  // node.
  // Values for the output_types and output_shapes node attributes are also
  // written if those attributes are defined in the OpDef.
  // `*output` contains a pointer to the output `Node`. It is guaranteed to be
  // non-null if the method returns with an OK status.
  // The returned Node pointer is owned by the backing Graph of GraphDefBuilder.
  template <class DatasetType>
  Status AddDataset(const DatasetType* dataset,
                    const std::vector<NodeBuilder::NodeOut>& inputs,
                    const std::vector<std::pair<StringPiece, AttrValue>>& attrs,
                    Node** output) {
    const string& op_type_name = dataset->op_name();
    std::unique_ptr<const GraphDefBuilder::Options> opts(
        new GraphDefBuilder::Options(b_->opts()));
    // TODO(srbs|mrry): Not all datasets have output_types and output_shapes
    // attributes defined. It will be nice to have a consistent pattern.
    bool has_output_types_attr = HasAttr(op_type_name, "output_types");
    bool has_output_shapes_attr = HasAttr(op_type_name, "output_shapes");
    if (has_output_shapes_attr) {
      opts.reset(new GraphDefBuilder::Options(
          opts->WithAttr("output_shapes", dataset->output_shapes())));
    }
    if (has_output_types_attr) {
      opts.reset(new GraphDefBuilder::Options(
          opts->WithAttr("output_types", dataset->output_dtypes())));
    }
    for (auto attr : attrs) {
      opts.reset(new GraphDefBuilder::Options(
          opts->WithAttr(attr.first, attr.second)));
    }
    if (opts->HaveError()) {
      return errors::Internal("AddDataset: Error building Options.");
    }
    NodeBuilder node_builder(opts->GetNameForOp(op_type_name), op_type_name,
                             opts->op_registry());
    for (auto node_out : inputs) {
      node_builder.Input(node_out);
    }
    *output = opts->FinalizeBuilder(&node_builder);
    if (*output == nullptr) {
      return errors::Internal("AddDataset: Failed to build ", op_type_name,
                              " op.");
    }
    return Status::OK();
  }

  // TODO(shivaniagrawal): Single method for AddDataset for
  // NodeOut/ArrraySlice<NodeOut>
  template <class DatasetType>
  Status AddDatasetWithInputAsList(const DatasetType* dataset,
                                   gtl::ArraySlice<NodeBuilder::NodeOut> input,
                                   Node** output) {
    const string& op_type_name = dataset->op_name();
    std::unique_ptr<const GraphDefBuilder::Options> opts(
        new GraphDefBuilder::Options(b_->opts()));
    bool has_output_types_attr = HasAttr(op_type_name, "output_types");
    bool has_output_shapes_attr = HasAttr(op_type_name, "output_shapes");
    if (has_output_shapes_attr) {
      opts.reset(new GraphDefBuilder::Options(
          opts->WithAttr("output_shapes", dataset->output_shapes())));
    }
    if (has_output_types_attr) {
      opts.reset(new GraphDefBuilder::Options(
          opts->WithAttr("output_types", dataset->output_dtypes())));
    }
    if (opts->HaveError()) {
      return errors::Internal("AddDataset: Error building Options.");
    }
    NodeBuilder node_builder(opts->GetNameForOp(op_type_name), op_type_name,
                             opts->op_registry());
    node_builder.Input(input);
    *output = opts->FinalizeBuilder(&node_builder);
    if (*output == nullptr) {
      return errors::Internal("AddDataset: Failed to build ", op_type_name,
                              " op.");
    }
    return Status::OK();
  }

  template <typename T>
  void BuildAttrValue(const T& value, AttrValue* attr) {
    SetAttrValue(value, attr);
  }

 private:
  void AddTensorInternal(const Tensor& val, Node** output) {
    *output = ops::SourceOp(
        "Const",
        b_->opts().WithAttr("dtype", val.dtype()).WithAttr("value", val));
  }

  bool HasAttr(const string& op_type_name, const string& attr_name) {
    const OpDef* op_def = nullptr;
    Status s = b_->opts().op_registry()->LookUpOpDef(op_type_name, &op_def);
    if (!s.ok() || op_def == nullptr) {
      return false;
    }
    for (auto attr : op_def->attr()) {
      if (attr.name() == attr_name) {
        return true;
      }
    }
    return false;
  }

  GraphDefBuilder* b_;
};

// A cut-down version of OpKernelContext for running computations in
// iterators. Note that we cannot simply use OpKernelContext here
// because we might run computation in an iterator whose lifetime is
// not nested within the lifetime of a single OpKernelContext
// (e.g. asynchronous prefetching).
//
// TODO(mrry): We will probably need to support more of
// OpKernelContext here. For example, should allocation be handled by
// the IteratorContext?
// TODO(mrry): We will need to fabricate step IDs for calls to ops
// that are not nested within a particular step.
// TODO(mrry): We're making some daring assumptions about the lifetime
// of the FunctionLibraryRuntime and runner passed in here. Once
// created, a FunctionLibraryRuntime should stay alive for the
// remainder of a session, so we copy the pointer. A runner will be
// deleted when the original step ends, but all existing runners only
// close over session-lifetime (or longer-lived) state, so we can make
// a copy of the function. There's nothing in the definition of either
// class to guarantee that what we are doing is safe. We should
// formalize the properties here.
class IteratorContext {
 public:
  struct Params {
    // Interface to operating system functionality.
    Env* env;

    // The step being executed.
    int64 step_id = 0;

    // Shared resources accessible by this iterator invocation.
    ResourceMgr* resource_manager = nullptr;

    // Function call support.
    std::function<void(std::function<void()>)> runner = nullptr;
  };

  explicit IteratorContext(Params params) : params_(std::move(params)) {}

  Env* env() const { return params_.env; }

  int64 step_id() const { return params_.step_id; }

  std::function<void(std::function<void()>)>* runner() {
    return &params_.runner;
  }

  ResourceMgr* resource_manager() const { return params_.resource_manager; }

 private:
  Params params_;
};

// Represents the current position in a range of outputs, where the
// range of outputs is typically represented by an `DatasetBase`,
// defined below.
class IteratorBase {
 public:
  virtual ~IteratorBase() {}

  // Gets the next output from the range that this iterator is traversing.
  //
  // If at least one output remains in this iterator's range, that
  // output will be stored in `*out_tensors` and `false` will be
  // stored in `*end_of_sequence`.
  //
  // If no more outputs remain in this iterator's range, `true` will
  // be stored in `*end_of_sequence`, and the content of
  // `*out_tensors` will be undefined.
  //
  // This method is thread-safe.
  //
  // TODO(mrry): Define `GetNextAsync()` or `GetNextManyAsync()`, and
  // potentially remove this method.
  virtual Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) = 0;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // iterator.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this iterator.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // Saves the state of this iterator.
  virtual Status Save(IteratorStateWriter* writer) {
    if (is_exhausted_) {
      LOG(INFO) << "Iterator exhausted.";
      return writer->WriteScalar(kIteratorExhausted, kIteratorExhausted);
    } else {
      return SaveInternal(writer);
    }
  }

  // Restores the state of this iterator.
  virtual Status Restore(OpKernelContext* ctx, IteratorStateReader* reader) {
    if (reader->Contains(kIteratorExhausted)) {
      LOG(INFO) << "Iterator exhausted. Nothing to restore.";
      is_exhausted_ = true;
      return Status::OK();
    } else {
      return RestoreInternal(ctx, reader);
    }
  }

  static const char kIteratorExhausted[];

 protected:
  // This is needed so that sub-classes of IteratorBase can call
  // `SaveInternal` on their parent iterators, e.g., in
  // `RepeatDataasetOp::Dataset`.
  Status SaveParent(IteratorStateWriter* writer,
                    const std::unique_ptr<IteratorBase>& parent) {
    return parent->SaveInternal(writer);
  }

  // This is needed so that sub-classes of IteratorBase can call
  // `RestoreInternal` on their parent iterators, e.g., in
  // `RepeatDataasetOp::Dataset`.
  Status RestoreParent(OpKernelContext* ctx, IteratorStateReader* reader,
                       const std::unique_ptr<IteratorBase>& parent) {
    return parent->RestoreInternal(ctx, reader);
  }

  // Saves the state of this iterator recursively.
  virtual Status SaveInternal(IteratorStateWriter* writer) {
    return errors::Unimplemented("SaveInternal");
  }

  // Restores the state of this iterator recursively.
  virtual Status RestoreInternal(OpKernelContext* ctx,
                                 IteratorStateReader* reader) {
    return errors::Unimplemented("RestoreInternal");
  }

  bool is_exhausted_ = false;  // Whether the iterator has been exhausted.
};

// Represents a (potentially infinite) range of outputs, where each
// output is a tuple of tensors.
class DatasetBase : public core::RefCounted {
 public:
  // Returns a new iterator for iterating over the range of elements in
  // this dataset.
  //
  // This method may be called multiple times on the same instance,
  // and the resulting iterators will have distinct state. Each
  // iterator will traverse all elements in this dataset from the
  // start.
  //
  // Ownership of the created iterator will be transferred to the caller.
  //
  // The prefix identifies the sequence of iterators leading up to the newly
  // created iterator.
  virtual std::unique_ptr<IteratorBase> MakeIterator(
      const string& prefix) const = 0;

  // Returns a vector of DataType values, representing the respective
  // element types of each tuple component in the outputs of this
  // dataset.
  virtual const DataTypeVector& output_dtypes() const = 0;

  // Returns a vector of tensor shapes, representing the respective
  // (and possibly partially defined) shapes of each tuple component
  // in the outputs of this dataset.
  virtual const std::vector<PartialTensorShape>& output_shapes() const = 0;

  // A human-readable debug string for this dataset.
  virtual string DebugString() = 0;

  // Serializes the dataset and writes it to the `writer`.
  virtual Status Save(IteratorStateWriter* writer) const {
    return errors::Unimplemented("DatasetBase::Save");
  }

 protected:
  // TODO(srbs): Ideally all graph related logic should reside in
  // GraphDatasetBase. However, that would require Datasets defined in all ops
  // to derive from GraphDatasetBase. Once that is done we can move
  // DatasetGraphDefBuilder and AsGraphDefInternal to GraphDatasetBase.
  class DatasetGraphDefBuilder : public GraphDefBuilderWrapper {
   public:
    DatasetGraphDefBuilder(GraphDefBuilder* b) : GraphDefBuilderWrapper(b) {}
    Status AddParentDataset(const DatasetBase* dataset, Node** output) {
      return dataset->AsGraphDefInternal(this, output);
    }
  };

  virtual Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                                    Node** node) const {
    return errors::Unimplemented("AsGraphDefInternal");
  }
};

// Base-class for datasets that are built by ops.
class GraphDatasetBase : public DatasetBase {
 public:
  GraphDatasetBase(OpKernelContext* ctx)
      : op_name_(ctx->op_kernel().type_string()) {}

  const string op_name() const { return op_name_; }

  Status Save(IteratorStateWriter* writer) const override {
    string serialized_graph_def;
    string output_node;
    TF_RETURN_IF_ERROR(Serialize(&serialized_graph_def, &output_node));
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(kDatasetGraphKey, serialized_graph_def));
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(kDatasetGraphOutputNodeKey, output_node));
    return Status::OK();
  }

  // Key for storing the Dataset graph in the serialized format.
  static const char kDatasetGraphKey[];

  // Key for storing the output node of the Dataset graph in the serialized
  // format.
  static const char kDatasetGraphOutputNodeKey[];

 private:
  Status Serialize(string* serialized_graph_def, string* output_node) const {
    GraphDefBuilder b;
    DatasetGraphDefBuilder db(&b);
    Node* node = nullptr;
    TF_RETURN_IF_ERROR(AsGraphDefInternal(&db, &node));
    *output_node = node->name();
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(b.ToGraphDef(&graph_def));
    graph_def.SerializeToString(serialized_graph_def);
    return Status::OK();
  }

  const string op_name_;
};

// Represents an iterator that is associated with a particular parent dataset.
template <class DatasetType>
class DatasetIterator : public IteratorBase {
 public:
  struct Params {
    // Owns one reference on the shared dataset resource.
    const DatasetType* dataset;

    // Identifies the sequence of iterators leading up to this iterator.
    const string prefix;
  };

  explicit DatasetIterator(const Params& params) : params_(params) {
    params_.dataset->Ref();
  }

  ~DatasetIterator() override { params_.dataset->Unref(); }

  // The dataset from which this iterator was created.
  const DatasetType* dataset() const { return params_.dataset; }

  // The sequence of iterators leading up to this iterator.
  const string prefix() const { return params_.prefix; }

  const DataTypeVector& output_dtypes() const override {
    return params_.dataset->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return params_.dataset->output_shapes();
  }

  Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                 bool* end_of_sequence) final {
    port::Tracing::TraceMe activity(params_.prefix);
    if (is_exhausted_) {
      *end_of_sequence = true;
      return Status::OK();
    }
    return GetNextInternal(ctx, out_tensors, end_of_sequence);
  }

  Status Save(IteratorStateWriter* writer) final {
    TF_RETURN_IF_ERROR(dataset()->Save(writer));
    return IteratorBase::Save(writer);
  }

 protected:
  // Internal implementation of GetNext that is wrapped in tracing logic.
  virtual Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) = 0;

  string full_name(const string& name) const {
    return strings::StrCat(prefix(), ":", name);
  }

 private:
  Params params_;
};

// Encapsulates the work required to plug a DatasetBase into the core TensorFlow
// graph execution engine.
class DatasetOpKernel : public OpKernel {
 public:
  DatasetOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) final;

 protected:
  // Subclasses should implement this method. It will be called during Compute
  // execution.
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase** output) = 0;

  template <typename T>
  Status ParseScalarArgument(OpKernelContext* ctx,
                             const StringPiece& argument_name, T* output) {
    const Tensor* argument_t;
    TF_RETURN_IF_ERROR(ctx->input(argument_name, &argument_t));
    if (!TensorShapeUtils::IsScalar(argument_t->shape())) {
      return errors::InvalidArgument(argument_name, " must be a scalar");
    }
    *output = argument_t->scalar<T>()();
    return Status::OK();
  }
};

// Encapsulates the work required to plug unary Datasets into the core
// TensorFlow graph execution engine.
class UnaryDatasetOpKernel : public DatasetOpKernel {
 public:
  UnaryDatasetOpKernel(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase** output) = 0;
};

// Encapsulates the work required to plug binary Datasets into the core
// TensorFlow graph execution engine.
class BinaryDatasetOpKernel : public DatasetOpKernel {
 public:
  BinaryDatasetOpKernel(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) final;
  virtual void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                           DatasetBase* another_input,
                           DatasetBase** output) = 0;
};

// Validates and extracts a `DatasetBase` object from `tensor`.
//
// `tensor` must have been written by a call to SetVariantTensorToDataset().
//
// The retrieved pointer is a borrowed reference to the dataset, which is owned
// by the tensor. The consumer must either acquire its own reference to the
// dataset by calling `(*out_dataset)->Ref()`, or ensure that `tensor` is not
// destroyed or mutated while the retrieved pointer is in use.
Status GetDatasetFromVariantTensor(const Tensor& tensor,
                                   DatasetBase** out_dataset);

// Stores a `DatasetBase` object in `tensor`.
//
// The ownership of `dataset` is transferred to `tensor`.
Status StoreDatasetInVariantTensor(DatasetBase* dataset, Tensor* tensor);

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_DATASET_H_
