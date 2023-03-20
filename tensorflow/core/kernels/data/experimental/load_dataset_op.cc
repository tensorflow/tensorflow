/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/experimental/load_dataset_op.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/data/service/snapshot/snapshot_reader.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const LoadDatasetOp::kCompression;
/* static */ constexpr const char* const LoadDatasetOp::kDatasetType;
/* static */ constexpr const char* const LoadDatasetOp::kOutputTypes;
/* static */ constexpr const char* const LoadDatasetOp::kOutputShapes;
/* static */ constexpr const char* const LoadDatasetOp::kPath;
/* static */ constexpr const char* const LoadDatasetOp::kReaderFunc;
/* static */ constexpr const char* const LoadDatasetOp::kReaderFuncOtherArgs;
/* static */ constexpr const char* const LoadDatasetOp::kReaderFuncTarguments;

class LoadDatasetOp::DatasetV1 : public DatasetBase {
 public:
  DatasetV1(OpKernelContext* ctx, const tstring& path,
            SnapshotMetadataRecord metadata, const std::string& compression,
            std::unique_ptr<CapturedFunction> captured_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        captured_func_(std::move(captured_func)),
        compression_(compression),
        metadata_(std::move(metadata)),
        output_types_(output_types),
        output_shapes_(output_shapes),
        path_(path) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return metadata_.num_elements();
  }

  Status CheckExternalState() const override {
    return captured_func_->CheckExternalState();
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->clear();
    return OkStatus();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* path_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(path_, &path_node));

    std::vector<Node*> reader_func_other_args;
    DataTypeVector reader_func_other_args_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(
        ctx, b, &reader_func_other_args, &reader_func_other_args_types));

    // Attr: compression
    AttrValue compression_attr;
    b->BuildAttrValue(compression_, &compression_attr);

    // Attr: reader_func
    AttrValue reader_func_attr;
    b->BuildAttrValue(captured_func_->func(), &reader_func_attr);

    AttrValue reader_func_arguments_types_attr;
    b->BuildAttrValue(reader_func_other_args_types,
                      &reader_func_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {std::make_pair(0, path_node)},         // Single tensor inputs.
        {std::make_pair(1, reader_func_other_args)},  // Tensor list inputs.
        {std::make_pair(kCompression, compression_attr),
         std::make_pair(kReaderFunc, reader_func_attr),
         std::make_pair(kReaderFuncTarguments,
                        reader_func_arguments_types_attr)},  // Attrs
        output));
    return OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<DatasetV1> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<DatasetV1>(params) {}

    ~Iterator() override {
      if (input_) {
        input_->Unref();
      }
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_));
      TF_RETURN_IF_ERROR(InitializeInput(ctx));
      return input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      return this->SaveInput(ctx, writer, input_impl_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      return this->RestoreInput(ctx, reader, input_impl_);
    }

   private:
    Status InitializeInput(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      auto run_dir = snapshot_util::RunDirectory(
          TranslateFileName(dataset()->path_), dataset()->metadata_.run_id());
      std::vector<std::string> snapshot_shard_dirs;
      TF_RETURN_IF_ERROR(ctx->env()->GetMatchingPaths(
          io::JoinPath(run_dir,
                       strings::Printf("%s%s", "*",
                                       snapshot_util::kShardDirectorySuffix)),
          &snapshot_shard_dirs));
      std::sort(snapshot_shard_dirs.begin(), snapshot_shard_dirs.end());

      DatasetBase* dataset_of_snapshot_files;
      TF_RETURN_IF_ERROR(snapshot_util::Reader::MakeNestedDataset(
          ctx->env(), snapshot_shard_dirs, dataset()->compression_,
          dataset()->metadata_.version(), dataset()->output_dtypes(),
          dataset()->output_shapes(), /*start_index=*/0,
          &dataset_of_snapshot_files));

      Tensor input_dataset_tensor(DT_VARIANT, TensorShape({}));
      TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(dataset_of_snapshot_files,
                                                     &input_dataset_tensor));

      std::vector<Tensor> reader_input;
      std::vector<Tensor> reader_output;
      reader_input.push_back(std::move(input_dataset_tensor));

      // NOTE: We intentionally ignore resource modeling outside GetNext().
      TF_RETURN_IF_ERROR(instantiated_captured_func_->Run(
          ctx, std::move(reader_input), &reader_output, /*node=*/nullptr));
      if (reader_output.size() != 1) {
        return errors::InvalidArgument(
            "reader_func returns more than one argument.");
      }
      TF_RETURN_IF_ERROR(
          GetDatasetFromVariantTensor(reader_output[0], &input_));
      // We need to take a reference here as we will use the input_ and
      // its iterator.
      input_->Ref();
      return OkStatus();
    }

    mutex mu_;
    DatasetBase* input_ TF_GUARDED_BY(mu_) = nullptr;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const std::unique_ptr<CapturedFunction> captured_func_;
  const std::string compression_;
  const SnapshotMetadataRecord metadata_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const tstring path_;
};

class LoadDatasetOp::DatasetV2 : public DatasetBase {
 public:
  DatasetV2(OpKernelContext* ctx, const tstring& path,
            DistributedSnapshotMetadata metadata,
            std::unique_ptr<CapturedFunction> captured_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        path_(path),
        metadata_(std::move(metadata)),
        captured_func_(std::move(captured_func)),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status CheckExternalState() const override { return OkStatus(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* path_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(path_, &path_node));

    std::vector<Node*> reader_func_other_args;
    DataTypeVector reader_func_other_args_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(
        ctx, b, &reader_func_other_args, &reader_func_other_args_types));

    // Attr: compression
    AttrValue compression_attr;
    b->BuildAttrValue(metadata_.compression(), &compression_attr);

    // Attr: reader_func
    AttrValue reader_func_attr;
    b->BuildAttrValue(captured_func_->func(), &reader_func_attr);

    AttrValue reader_func_arguments_types_attr;
    b->BuildAttrValue(reader_func_other_args_types,
                      &reader_func_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {std::make_pair(0, path_node)},         // Single tensor inputs.
        {std::make_pair(1, reader_func_other_args)},  // Tensor list inputs.
        {std::make_pair(kCompression, compression_attr),
         std::make_pair(kReaderFunc, reader_func_attr),
         std::make_pair(kReaderFuncTarguments,
                        reader_func_arguments_types_attr)},  // Attrs
        output));
    return OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<DatasetV2> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<DatasetV2>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func;
      TF_RETURN_IF_ERROR(dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func));

      SnapshotReaderParams params{dataset()->path_, dataset()->metadata_,
                                  dataset()->output_types_,
                                  dataset()->output_shapes_, ctx->env()};
      core::RefCountPtr<DatasetBase> input;
      TF_RETURN_IF_ERROR(MakeSnapshotReaderDataset(
          params, *instantiated_captured_func, ctx, &input));
      return input->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return errors::Unimplemented("Checkpointing is not currently supported.");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return errors::Unimplemented("Checkpointing is not currently supported.");
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  };

  const tstring path_;
  const DistributedSnapshotMetadata metadata_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

LoadDatasetOp::LoadDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCompression, &compression_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kReaderFunc, /*params=*/{},
                                               &func_metadata_));
}

void LoadDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  tstring path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kPath, &path));

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(ctx, func_metadata_, kReaderFuncOtherArgs,
                                    &captured_func));

  experimental::DistributedSnapshotMetadata distributed_metadata;
  if (Status s = ReadTextProto(ctx->env(), SnapshotMetadataFilePath(path),
                               &distributed_metadata);
      s.ok()) {
    *output =
        new DatasetV2(ctx, path, std::move(distributed_metadata),
                      std::move(captured_func), output_types_, output_shapes_);
    return;
  }

  bool metadata_file_exists;
  experimental::SnapshotMetadataRecord nondistributed_metadata;
  OP_REQUIRES_OK(ctx, snapshot_util::ReadMetadataFile(ctx->env(), path,
                                                      &nondistributed_metadata,
                                                      &metadata_file_exists));
  OP_REQUIRES(ctx, metadata_file_exists,
              errors::NotFound("Could not find metadata file [", path, "]"));
  *output =
      new DatasetV1(ctx, path, std::move(nondistributed_metadata), compression_,
                    std::move(captured_func), output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("LoadDataset").Device(DEVICE_CPU), LoadDatasetOp);
}  // namespace

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
