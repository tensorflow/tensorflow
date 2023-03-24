/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/snapshot_reader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/refcount.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char* const kCompression = "compression";

constexpr int64_t kTFRecordReaderOutputBufferSize = 256 << 20;  // 256MB

// A reader dataset is responsible for reading one chunk file.
// TODO(b/250921378): Merge this with `snapshot_util::Reader::Dataset`.
class ReaderDatasetOp : public DatasetOpKernel {
 public:
  explicit ReaderDatasetOp(OpKernelConstruction* ctx);
  class Dataset;

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::string compression_;
};

class ReaderDatasetOp::Dataset : public DatasetBase {
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

  std::string DebugString() const override { return "SnapshotDatasetReaderV2"; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return OkStatus();
  }

  Status CheckExternalState() const override { return OkStatus(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    std::vector<Node*> inputs;
    Node* chunk_file = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(chunk_file_, &chunk_file));
    inputs.push_back(chunk_file);

    std::vector<std::pair<StringPiece, AttrValue>> attrs;
    AttrValue compression;
    b->BuildAttrValue(compression_, &compression);
    attrs.push_back({kCompression, compression});
    return b->AddDataset(this, inputs, attrs, output);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      reader_ = std::make_unique<snapshot_util::TFRecordReader>(
          dataset()->chunk_file_, dataset()->compression_, dataset()->dtypes_,
          kTFRecordReaderOutputBufferSize);
      return reader_->Initialize(ctx->env());
    }

   protected:
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      *end_of_sequence = false;
      Status s = reader_->ReadTensors(out_tensors);
      if (errors::IsOutOfRange(s)) {
        *end_of_sequence = true;
        return OkStatus();
      }
      return s;
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented(
          "TODO(b/250921378: Support save/load for tf.data distributed "
          "snapshot reader.)");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented(
          "TODO(b/250921378: Support save/load for tf.data distributed "
          "snapshot reader.)");
    }

   private:
    std::unique_ptr<snapshot_util::TFRecordReader> reader_;
  };

  const tstring chunk_file_;
  const tstring compression_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
};

Status MakeNestedDataset(const SnapshotReaderParams& params,
                         DatasetBase** output) {
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> chunk_files,
      GetChildren(params.CommittedChunksDirectory(), params.env));

  std::vector<DatasetBase*> datasets;
  datasets.reserve(chunk_files.size());
  for (int64_t i = 0; i < chunk_files.size(); ++i) {
    std::string chunk_file_path =
        tsl::io::JoinPath(params.CommittedChunksDirectory(), chunk_files[i]);
    datasets.push_back(new ReaderDatasetOp::Dataset(
        DatasetContext(DatasetContext::Params(
            {"SnapshotDatasetReaderV2",
             strings::StrCat("SnapshotDatasetReaderV2/_", i)})),
        chunk_file_path, params.metadata.compression(), params.dtypes,
        params.shapes));
    datasets.back()->Initialize(/*metadata=*/{});
  }
  snapshot_util::Reader::MakeNestedDataset(datasets, output);
  return OkStatus();
}

}  // namespace

StatusOr<core::RefCountPtr<DatasetBase>> MakeSnapshotReaderDataset(
    const SnapshotReaderParams& params,
    InstantiatedCapturedFunction& instantiated_captured_func,
    IteratorContext* ctx) {
  DatasetBase* dataset_of_snapshot_files;
  TF_RETURN_IF_ERROR(MakeNestedDataset(params, &dataset_of_snapshot_files));

  Tensor input_dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(dataset_of_snapshot_files,
                                                 &input_dataset_tensor));

  std::vector<Tensor> reader_input;
  std::vector<Tensor> reader_output;
  reader_input.push_back(std::move(input_dataset_tensor));

  // NOTE: We intentionally ignore resource modeling outside GetNext().
  TF_RETURN_IF_ERROR(instantiated_captured_func.Run(
      ctx, std::move(reader_input), &reader_output, /*node=*/nullptr));
  if (reader_output.size() != 1) {
    return errors::InvalidArgument(
        "reader_func in tf.data.Dataset.load is expected to return one "
        "argument. Got ",
        reader_output.size(), ".");
  }
  DatasetBase* output_dataset = nullptr;
  TF_RETURN_IF_ERROR(
      GetDatasetFromVariantTensor(reader_output[0], &output_dataset));
  output_dataset->Ref();
  return core::RefCountPtr<DatasetBase>(output_dataset);
}

}  // namespace data
}  // namespace tensorflow
