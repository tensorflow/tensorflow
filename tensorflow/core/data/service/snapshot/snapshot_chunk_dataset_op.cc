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
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char* const kChunkFile = "chunk_file";
constexpr const char* const kCompression = "compression";
constexpr const char* const kStartIndex = "start_index";
constexpr const char* const kOutputTypes = "output_types";
constexpr const char* const kOutputShapes = "output_shapes";
constexpr const char* const kSnapshotChunkDataset = "SnapshotChunkDataset";

constexpr int64_t kTFRecordReaderOutputBufferSize = 512 << 20;  // 512MB

absl::string_view GetSnapshotPath(absl::string_view chunk_file) {
  // Snapshot chunks are placed in snapshot_path/chunks/chunk_x.
  absl::string_view chunk_dir = tsl::io::Dirname(chunk_file);
  return tsl::io::Dirname(chunk_dir);
}

// A reader dataset is responsible for reading one chunk file of a snapshot.
// TODO(b/250921378): Merge this with `snapshot_util::Reader::Dataset`.
class SnapshotChunkDatasetOp : public DatasetOpKernel {
 public:
  explicit SnapshotChunkDatasetOp(OpKernelConstruction* ctx);
  class Dataset;

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::string compression_;
};

class SnapshotChunkDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(DatasetContext&& ctx, const std::string& chunk_file,
          const std::string& compression, const DataTypeVector& dtypes,
          const std::vector<PartialTensorShape>& shapes)
      : DatasetBase(std::move(ctx)),
        chunk_file_(chunk_file),
        compression_(compression),
        dtypes_(dtypes),
        shapes_(shapes) {}

  const DataTypeVector& output_dtypes() const override { return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return shapes_;
  }

  std::string DebugString() const override { return "SnapshotChunkDataset"; }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override { return absl::OkStatus(); }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* chunk_file = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(chunk_file_, &chunk_file));

    AttrValue compression;
    b->BuildAttrValue(compression_, &compression);

    return b->AddDataset(this,
                         /*inputs=*/
                         {std::make_pair(0, chunk_file)},
                         /*list_inputs=*/{},
                         /*attrs=*/
                         {{kCompression, compression}},
                         /*use_dataset_name=*/true, output);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    ~Iterator() override { RecordBytesRead(); }

    absl::Status Initialize(IteratorContext* ctx) override {
      reader_ = std::make_unique<snapshot_util::TFRecordReader>(
          TranslateFileName(dataset()->chunk_file_), dataset()->compression_,
          dataset()->dtypes_, kTFRecordReaderOutputBufferSize);
      return reader_->Initialize(ctx->env());
    }

   protected:
    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      *end_of_sequence = false;
      absl::Status status = reader_->ReadTensors(out_tensors);
      if (absl::IsOutOfRange(status)) {
        *end_of_sequence = true;
        return absl::OkStatus();
      }
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          status,
          " Failed to read tf.data snapshot file: ", dataset()->chunk_file_);
      ++start_index_;
      return status;
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kStartIndex), start_index_));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kStartIndex), &start_index_));
      TF_RETURN_IF_ERROR(Initialize(ctx));
      return AdvanceToStartIndex(ctx);
    }

   private:
    // TODO(b/250921378): Optimize this to not parse every single element. We
    // may consider switching the data format to ArrayRecords so we can use the
    // index to jump straight to the starting record.
    absl::Status AdvanceToStartIndex(IteratorContext* ctx) {
      for (int64_t i = 0; i < start_index_; ++i) {
        std::vector<Tensor> unused;
        TF_RETURN_IF_ERROR(reader_->ReadTensors(&unused));
      }
      return absl::OkStatus();
    }

    void RecordBytesRead() {
      uint64_t bytes_read = reader_->BytesRead();
      metrics::GetTFDataBytesReadCounter(kSnapshotChunkDataset)
          ->IncrementBy(bytes_read);
    }

    std::unique_ptr<snapshot_util::TFRecordReader> reader_;
    int64_t start_index_ = 0;
  };

  const tstring chunk_file_;
  const tstring compression_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
};

SnapshotChunkDatasetOp::SnapshotChunkDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCompression, &compression_));
}

void SnapshotChunkDatasetOp::MakeDataset(OpKernelContext* ctx,
                                         DatasetBase** output) {
  tstring chunk_file;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kChunkFile, &chunk_file));

  *output = new SnapshotChunkDatasetOp::Dataset(DatasetContext(ctx), chunk_file,
                                                compression_, output_types_,
                                                output_shapes_);
  metrics::RecordTFDataServiceSnapshotOp(
      std::string(GetSnapshotPath(chunk_file)), kSnapshotChunkDataset);
}

REGISTER_KERNEL_BUILDER(Name(kSnapshotChunkDataset).Device(DEVICE_CPU),
                        SnapshotChunkDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
