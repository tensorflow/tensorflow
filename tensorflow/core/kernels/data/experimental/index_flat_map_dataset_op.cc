/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kDatasetType[] = "IndexFlatMap";
constexpr const char kIndexFlatMapDataset[] = "IndexFlatMapDataset";
constexpr const char kMapFn[] = "map_func";
constexpr const char kMapFuncTargs[] = "Tmap_func_args";
constexpr const char kMapFuncOtherArgs[] = "map_func_other_args";
constexpr const char kIndexMapFn[] = "index_map_func";
constexpr const char kIndexMapFuncTargs[] = "Tindex_map_func_args";
constexpr const char kIndexMapFuncOtherArgs[] = "index_map_func_other_args";
constexpr const char kOutputCardinality[] = "output_cardinality";
constexpr const char kOutputTypes[] = "output_types";
constexpr const char kOutputShapes[] = "output_shapes";
constexpr const char kElementCount[] = "element_count";
constexpr const char kInputElementCount[] = "input_element_count";
constexpr const char kInputUnflattenedTensors[] = "input_unflattened_tensors";
constexpr const char kInputUnflattenedTensorsSize[] =
    "input_unflattened_tensors_size";

std::string ToDebugString(const std::vector<Tensor>& tensors) {
  std::vector<std::string> tensor_strs;
  tensor_strs.reserve(tensors.size());
  for (const Tensor& tensor : tensors) {
    tensor_strs.push_back(tensor.DebugString());
  }
  return absl::StrCat("{", absl::StrJoin(tensor_strs, ", "), "}");
}

absl::StatusOr<size_t> GetValue(const Tensor& tensor) {
  switch (tensor.dtype()) {
    case DT_UINT64:
      return tensor.scalar<uint64_t>()();
    case DT_UINT32:
      return tensor.scalar<uint32_t>()();
    case DT_INT64:
      return tensor.scalar<int64_t>()();
    case DT_INT32:
      return tensor.scalar<int32_t>()();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "The `index_map_func` for `index_flat_map` is expected to return two "
          "int32/int64 values representing the element index and an offset "
          "within the element. Got: ",
          tensor.DebugString()));
  }
}

// Returns the `offset`-th element from `tensors`.
absl::StatusOr<std::vector<Tensor>> GetSlice(const std::vector<Tensor>& tensors,
                                             size_t offset) {
  if (tensors.size() > 1) {
    if (offset >= tensors.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`index_flat_map` got invalid `index_map_func` which returns offset ",
          offset, ", but the input element has ", tensors.size(),
          " elements: ", ToDebugString(tensors)));
    }
    return std::vector<Tensor>{tensors[offset]};
  }

  std::vector<Tensor> result;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].dims() == 0) {  // Scalar.
      result.push_back(tensors[i]);
      continue;
    }
    if (offset > tensors[i].dim_size(0)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "`index_flat_map` got invalid `index_map_func` which returns offset ",
          offset, ", but the input element has ", tensors[i].dim_size(0),
          " elements: ", tensors[i].DebugString()));
    }
    result.push_back(MaybeCopySubSlice(tensors[i], offset));
  }
  return result;
}

class IndexFlatMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit IndexFlatMapDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  std::shared_ptr<FunctionMetadata> map_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> index_map_func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class IndexFlatMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_map_func,
          std::unique_ptr<CapturedFunction> captured_index_map_func,
          const int64_t output_cardinality, const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_map_func_(std::move(captured_map_func)),
        captured_index_map_func_(std::move(captured_index_map_func)),
        output_cardinality_(output_cardinality),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  std::string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return output_cardinality_;
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

  absl::Status RandomIndexingCompatible() const override {
    return absl::OkStatus();
  }

 protected:
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override;

  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    std::vector<Node*> map_func_other_args;
    DataTypeVector map_func_other_args_types;
    TF_RETURN_IF_ERROR(captured_map_func_->AddToGraph(
        ctx, b, &map_func_other_args, &map_func_other_args_types));

    std::vector<Node*> index_map_func_other_args;
    DataTypeVector index_map_func_other_args_types;
    TF_RETURN_IF_ERROR(captured_index_map_func_->AddToGraph(
        ctx, b, &index_map_func_other_args, &index_map_func_other_args_types));

    Node* output_cardinality;
    TF_RETURN_IF_ERROR(b->AddScalar(output_cardinality_, &output_cardinality));

    AttrValue map_func_attr;
    b->BuildAttrValue(captured_map_func_->func(), &map_func_attr);

    AttrValue map_func_arguments_types_attr;
    b->BuildAttrValue(map_func_other_args_types,
                      &map_func_arguments_types_attr);

    AttrValue index_map_func_attr;
    b->BuildAttrValue(captured_index_map_func_->func(), &index_map_func_attr);

    AttrValue index_map_func_arguments_types_attr;
    b->BuildAttrValue(index_map_func_other_args_types,
                      &index_map_func_arguments_types_attr);

    return b->AddDataset(
        this,
        /*inputs=*/
        {std::make_pair(0, input_graph_node),
         std::make_pair(3, output_cardinality)},
        /*list_inputs=*/
        {std::make_pair(1, map_func_other_args),
         std::make_pair(2, index_map_func_other_args)},
        /*attrs=*/
        {{kMapFn, map_func_attr},
         {kMapFuncTargs, map_func_arguments_types_attr},
         {kIndexMapFn, index_map_func_attr},
         {kIndexMapFuncTargs, index_map_func_arguments_types_attr}},
        output);
  }

 private:
  class Iterator;
  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_map_func_;
  const std::unique_ptr<CapturedFunction> captured_index_map_func_;
  const int64_t output_cardinality_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

class IndexFlatMapDatasetOp::Dataset::Iterator
    : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const Params& params) : DatasetIterator<Dataset>(params) {}

  absl::Status Initialize(IteratorContext* ctx) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);
    TF_RETURN_IF_ERROR(
        dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
    TF_RETURN_IF_ERROR(dataset()->captured_map_func_->Instantiate(
        ctx, &instantiated_map_func_));
    TF_RETURN_IF_ERROR(dataset()->captured_index_map_func_->Instantiate(
        ctx, &instantiated_index_map_func_));
    return absl::OkStatus();
  }

  absl::Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    if (ctx->index_mapper()) {
      return Get(ctx, out_tensors, end_of_sequence);
    }

    absl::MutexLock l(&mu_);
    absl::StatusOr<std::tuple<size_t, size_t>> next_input_index_and_offset =
        GetUnflattenedIndex(ctx, element_count_);
    TF_RETURN_IF_ERROR(next_input_index_and_offset.status());
    const auto [next_input_index, offset] =
        *std::move(next_input_index_and_offset);
    // When all the values of the current input element have been read,
    // advances to the next input element. Otherwise, returns an element from
    // the current `input_unflattened_tensors_`.
    if (next_input_index > input_element_count_ ||
        input_unflattened_tensors_.empty()) {
      input_unflattened_tensors_.clear();
      TF_RETURN_IF_ERROR(GetMappedTensorsFromInput(
          ctx, &input_unflattened_tensors_, end_of_sequence));
      if (*end_of_sequence) {
        return absl::OkStatus();
      }
      input_element_count_ = next_input_index;
    }
    TF_ASSIGN_OR_RETURN(*out_tensors,
                        GetSlice(input_unflattened_tensors_, offset));
    ++element_count_;
    return absl::OkStatus();
  }

  absl::Status Get(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                   bool* end_of_sequence) ABSL_LOCKS_EXCLUDED(mu_) {
    const int64_t cardinality = dataset()->Cardinality();
    if (cardinality < 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Global shuffling requires finite cardinality. Got cardinality ",
          cardinality, " for dataset ", dataset()->DebugString(), "."));
    }

    absl::MutexLock l(&mu_);
    size_t offset = 0;
    IteratorContext ctx_with_index_mapper =
        GetContextWithIndexMapper(ctx, offset);
    std::vector<Tensor> mapped_tensors;
    TF_RETURN_IF_ERROR(GetMappedTensorsFromInput(
        &ctx_with_index_mapper, &mapped_tensors, end_of_sequence));
    ctx->MergeCheckpoint(ctx_with_index_mapper.checkpoint());
    if (*end_of_sequence) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(*out_tensors, GetSlice(mapped_tensors, offset));
    return absl::OkStatus();
  }

  absl::Status GetMappedTensorsFromInput(IteratorContext* ctx,
                                         std::vector<Tensor>* mapped_tensors,
                                         bool* end_of_sequence)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    std::vector<Tensor> input_tensors;
    TF_RETURN_IF_ERROR(
        input_impl_->GetNext(ctx, &input_tensors, end_of_sequence));
    if (*end_of_sequence) {
      return absl::OkStatus();
    }
    return instantiated_map_func_->Run(ctx, {std::move(input_tensors)},
                                       mapped_tensors);
  }

  IteratorContext GetContextWithIndexMapper(IteratorContext* ctx,
                                            size_t& offset) const {
    IteratorContext::Params params(ctx);
    params.index_mapper = GetFlatMapIndexMapper(ctx, offset);
    return IteratorContext(params);
  }

  IndexMapperFn GetFlatMapIndexMapper(IteratorContext* ctx,
                                      size_t& offset) const {
    return [this, ctx,
            &offset](size_t element_position) -> absl::StatusOr<size_t> {
      if (ctx->index_mapper()) {
        TF_ASSIGN_OR_RETURN(element_position,
                            ctx->index_mapper()(element_position));
      }
      absl::StatusOr<std::tuple<size_t, size_t>> unflattened_index =
          GetUnflattenedIndex(ctx, element_position);
      TF_RETURN_IF_ERROR(unflattened_index.status());
      offset = std::get<1>(*unflattened_index);
      return std::get<0>(*unflattened_index);
    };
  }

  // Given an index in the flattened dataset, returns a tuple of
  // (element index, offset within element) in the unflattend dataset.
  absl::StatusOr<std::tuple<size_t, size_t>> GetUnflattenedIndex(
      IteratorContext* ctx, size_t flattened_index) const {
    Tensor flattened_index_tensor(ctx->allocator({}), DT_INT64,
                                  TensorShape({}));
    flattened_index_tensor.scalar<int64_t>()() = flattened_index;

    std::vector<Tensor> unflattened_index;
    TF_RETURN_IF_ERROR(instantiated_index_map_func_->Run(
        ctx, {std::move(flattened_index_tensor)}, &unflattened_index));
    if (unflattened_index.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The `index_map_fn` for `index_flat_map` is expected to return two "
          "int values representing the element index and an offset within the "
          "element. Got: ",
          ToDebugString(unflattened_index)));
    }
    TF_ASSIGN_OR_RETURN(size_t element_index, GetValue(unflattened_index[0]));
    TF_ASSIGN_OR_RETURN(size_t offset, GetValue(unflattened_index[1]));
    return std::tuple<size_t, size_t>{element_index, offset};
  }

  bool SymbolicCheckpointCompatible() const override { return true; }

  absl::Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(prefix(), kElementCount, element_count_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kInputElementCount,
                                           input_element_count_));
    TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(),
                                           kInputUnflattenedTensorsSize,
                                           input_unflattened_tensors_.size()));
    for (int64_t i = 0; i < input_unflattened_tensors_.size(); ++i) {
      TF_RETURN_IF_ERROR(writer->WriteTensor(
          prefix(), absl::StrCat(kInputUnflattenedTensors, "[", i, "]"),
          input_unflattened_tensors_[i]));
    }
    return SaveInput(ctx, writer, input_impl_);
  }

  absl::Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);
    if (ctx->restored_element_count().has_value()) {
      return RestoreInput(ctx, reader, input_impl_);
    }
    TF_RETURN_IF_ERROR(
        reader->ReadScalar(prefix(), kElementCount, &element_count_));
    TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kInputElementCount,
                                          &input_element_count_));

    int64_t input_unflattened_tensors_size = 0;
    TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(),
                                          kInputUnflattenedTensorsSize,
                                          &input_unflattened_tensors_size));
    input_unflattened_tensors_.clear();
    input_unflattened_tensors_.reserve(input_unflattened_tensors_size);
    for (int64_t i = 0; i < input_unflattened_tensors_size; ++i) {
      Tensor tensor;
      TF_RETURN_IF_ERROR(reader->ReadTensor(
          prefix(), absl::StrCat(kInputUnflattenedTensors, "[", i, "]"),
          &tensor));
      input_unflattened_tensors_.push_back(std::move(tensor));
    }
    return RestoreInput(ctx, reader, input_impl_);
  }

 private:
  mutable absl::Mutex mu_;
  std::unique_ptr<IteratorBase> input_impl_ ABSL_GUARDED_BY(mu_);

  // Tracks the input element. When not using global shuffling, these record the
  // current element count, the element count of the input iterator, and the
  // current output of the input iterator.
  int64_t element_count_ ABSL_GUARDED_BY(mu_) = 0;
  int64_t input_element_count_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<Tensor> input_unflattened_tensors_ ABSL_GUARDED_BY(mu_);

  std::unique_ptr<InstantiatedCapturedFunction> instantiated_map_func_;
  std::unique_ptr<InstantiatedCapturedFunction> instantiated_index_map_func_;
};

IndexFlatMapDatasetOp::IndexFlatMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kMapFn, /*params=*/{},
                                               &map_func_metadata_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kIndexMapFn, /*params=*/{},
                                               &index_map_func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void IndexFlatMapDatasetOp::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase* input,
                                        DatasetBase** output) {
  std::unique_ptr<CapturedFunction> captured_map_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(ctx, map_func_metadata_, kMapFuncOtherArgs,
                                    &captured_map_func));

  std::unique_ptr<CapturedFunction> captured_index_map_func;
  OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, index_map_func_metadata_,
                                               kIndexMapFuncOtherArgs,
                                               &captured_index_map_func));

  int64_t output_cardinality;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kOutputCardinality, &output_cardinality));

  *output = new Dataset(ctx, input, std::move(captured_map_func),
                        std::move(captured_index_map_func), output_cardinality,
                        output_types_, output_shapes_);
}

std::unique_ptr<IteratorBase>
IndexFlatMapDatasetOp::Dataset::MakeIteratorInternal(
    const std::string& prefix) const {
  return std::make_unique<IndexFlatMapDatasetOp::Dataset::Iterator>(
      Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)});
}

REGISTER_KERNEL_BUILDER(Name(kIndexFlatMapDataset).Device(DEVICE_CPU),
                        IndexFlatMapDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
