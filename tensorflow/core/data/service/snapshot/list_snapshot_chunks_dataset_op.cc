/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_chunk_provider.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kListSnapshotChunksDataset[] = "ListSnapshotChunksDataset";
constexpr const char kSnapshotPath[] = "snapshot_path";

class ListSnapshotChunksDatasetOp : public DatasetOpKernel {
 public:
  explicit ListSnapshotChunksDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class ListSnapshotChunksDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, tsl::tstring snapshot_path,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        snapshot_path_(std::move(snapshot_path)),
        output_types_(output_types),
        output_shapes_(output_shapes),
        env_(ctx->env()) {}

  absl::string_view snapshot_path() const { return snapshot_path_; }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return SnapshotChunksCardinality(snapshot_path(), env_);
  }

  absl::Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                      split_providers) const override {
    split_providers->push_back(
        std::make_unique<SnapshotChunkProvider>(snapshot_path_, env_));
    return absl::OkStatus();
  }

  std::string DebugString() const override {
    return name_utils::DatasetDebugString(kListSnapshotChunksDataset);
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override { return absl::OkStatus(); }

 protected:
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override;

  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* snapshot_path = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(snapshot_path_, &snapshot_path));
    return b->AddDataset(this, /*inputs=*/{snapshot_path}, output);
  }

 private:
  class Iterator;

  const tsl::tstring snapshot_path_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  tsl::Env* const env_;
};

class ListSnapshotChunksDatasetOp::Dataset::Iterator
    : public DatasetIterator<ListSnapshotChunksDatasetOp::Dataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<ListSnapshotChunksDatasetOp::Dataset>(params) {}

  absl::Status Initialize(IteratorContext* ctx) override {
    if (ctx->split_providers().empty()) {
      split_provider_ = std::make_shared<SnapshotChunkProvider>(
          dataset()->snapshot_path(), ctx->env());
    } else {
      TF_ASSIGN_OR_RETURN(split_provider_,
                          GetSingleSplitProvider(ctx, dataset()));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
    Tensor split;
    TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, end_of_sequence));
    if (*end_of_sequence) {
      return absl::OkStatus();
    }
    out_tensors->push_back(std::move(split));
    return absl::OkStatus();
  }

  absl::Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override {
    return split_provider_->Save(
        [&](const std::string& key) { return full_name(key); }, writer);
  }

  absl::Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
    return split_provider_->Restore(
        [&](const std::string& key) { return full_name(key); }, reader);
  }

  std::shared_ptr<SplitProvider> split_provider_;
};

ListSnapshotChunksDatasetOp::ListSnapshotChunksDatasetOp(
    OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
}

void ListSnapshotChunksDatasetOp::MakeDataset(OpKernelContext* ctx,
                                              DatasetBase** output) {
  tsl::tstring snapshot_path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kSnapshotPath, &snapshot_path));
  OP_REQUIRES(ctx, !snapshot_path.empty(),
              absl::InvalidArgumentError(
                  "snapshot_path is required to list snapshot chunks."));
  *output = new ListSnapshotChunksDatasetOp::Dataset(
      ctx, std::move(snapshot_path), output_types_, output_shapes_);
}

std::unique_ptr<IteratorBase>
ListSnapshotChunksDatasetOp::Dataset::MakeIteratorInternal(
    const std::string& prefix) const {
  return std::make_unique<ListSnapshotChunksDatasetOp::Dataset::Iterator>(
      ListSnapshotChunksDatasetOp::Dataset::Iterator::Params{
          this,
          name_utils::IteratorPrefix(kListSnapshotChunksDataset, prefix)});
}

REGISTER_KERNEL_BUILDER(Name(kListSnapshotChunksDataset).Device(DEVICE_CPU),
                        ListSnapshotChunksDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
