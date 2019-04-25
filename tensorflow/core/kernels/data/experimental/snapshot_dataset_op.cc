/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/protobuf/data/experimental/snapshot.pb.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace {

enum SnapshotState { STATE_READ = 0, STATE_WRITE = 1, STATE_PASSTHROUGH = 2 };

const uint64 ONE_DAY_IN_MICROSECONDS = 24L * 60L * 60L * 1e6L;

class SnapshotDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SnapshotDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    string path;

    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "path", &path));

    GraphDef graph_def;
    OP_REQUIRES_OK(ctx, AsGraphDef(ctx, input, &graph_def));
    string graph_fp = strings::StrCat(strings::Hex(
        Fingerprint64(graph_def.SerializeAsString()), strings::kZeroPad4));

    *output = new Dataset(ctx, input, path, graph_fp);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, const string& path,
            const string& graph_fp)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          path_(path),
          graph_fp_(graph_fp) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Snapshot")});
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "SnapshotDatasetOp::Dataset"; }

    int64 Cardinality() const override { return input_->Cardinality(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* path = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(path_, &path));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node, path}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        run_id_ =
            strings::StrCat(strings::Hex(random::New64(), strings::kZeroPad4));
        fp_path_ = absl::StrCat(dataset()->path_, "/", dataset()->graph_fp_);
        state_ = DetermineOpState();

        return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // TODO(frankchn,rohanj): Implement the actual read/write of snapshots.

        switch (state_) {
          case STATE_READ:
            return GetNextInternalRead(ctx, out_tensors, end_of_sequence);
          case STATE_WRITE:
            return GetNextInternalWrite(ctx, out_tensors, end_of_sequence);
          case STATE_PASSTHROUGH:
          default:
            return GetNextInternalPassthrough(ctx, out_tensors,
                                              end_of_sequence);
        }
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        // TODO(frankchn): Make save iterators work
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        // TODO(frankchn): Make iterator restores work
        return Status::OK();
      }

     private:
      Status GetNextInternalRead(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) {
        return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
      }

      Status GetNextInternalWrite(IteratorContext* ctx,
                                  std::vector<Tensor>* out_tensors,
                                  bool* end_of_sequence) {
        return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
      }

      Status GetNextInternalPassthrough(IteratorContext* ctx,
                                        std::vector<Tensor>* out_tensors,
                                        bool* end_of_sequence) {
        return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
      }

      SnapshotState DetermineOpState() {
        experimental::SnapshotMetadataRecord metadata;
        Status s = ReadMetadataFile(&metadata);

        if (!s.ok()) {
          // File not found, or otherwise not readable.
          return STATE_WRITE;
        }

        if (metadata.finalized()) {
          // File found, snapshot has been finalized.
          return STATE_READ;
        }

        if (metadata.creation_timestamp() >=
            Env::Default()->NowMicros() - ONE_DAY_IN_MICROSECONDS) {
          // TODO(frankchn): Make this timestamp configurable.
          // Someone else is already writing and time has not expired.
          return STATE_PASSTHROUGH;
        } else {
          // Time has expired, we write regardless.
          return STATE_WRITE;
        }
      }

      Status ReadMetadataFile(experimental::SnapshotMetadataRecord* metadata) {
        string metadata_file_path =
            absl::StrCat(fp_path_, "/snapshot.metadata");
        TF_RETURN_IF_ERROR(Env::Default()->FileExists(metadata_file_path));

        std::unique_ptr<RandomAccessFile> file;
        TF_CHECK_OK(
            Env::Default()->NewRandomAccessFile(metadata_file_path, &file));

        string record_bytes;
        auto reader = absl::make_unique<io::RecordReader>(file.get());
        uint64 offset = 0;
        TF_CHECK_OK(reader->ReadRecord(&offset, &record_bytes));

        metadata->ParseFromString(record_bytes);
        return Status::OK();
      }

      string run_id_;
      string fp_path_;
      SnapshotState state_;

      std::unique_ptr<IteratorBase> input_impl_;
    };

    const DatasetBase* const input_;
    const string path_;
    const string graph_fp_;
  };

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("SnapshotDataset").Device(DEVICE_CPU),
                        SnapshotDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
