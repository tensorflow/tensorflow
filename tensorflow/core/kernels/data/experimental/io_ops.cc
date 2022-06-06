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

#include "tensorflow/core/kernels/data/experimental/io_ops.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/data/snapshot_utils.h"
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

/* static */ constexpr const char* const SaveDatasetOp::kCompression;
/* static */ constexpr const char* const SaveDatasetOp::kPath;
/* static */ constexpr const char* const SaveDatasetOp::kShardFunc;
/* static */ constexpr const char* const SaveDatasetOp::kShardFuncOtherArgs;
/* static */ constexpr const char* const SaveDatasetOp::kUseShardFunc;
/* static */ constexpr const int SaveDatasetOp::kFileFormatVersion;
/* static */ constexpr const char* const SaveDatasetV2Op::kInputDataset;
/* static */ constexpr const char* const SaveDatasetV2Op::kPath;
/* static */ constexpr const char* const SaveDatasetV2Op::kCompression;
/* static */ constexpr const char* const SaveDatasetV2Op::kDatasetType;
/* static */ constexpr const char* const SaveDatasetV2Op::kOutputTypes;
/* static */ constexpr const char* const SaveDatasetV2Op::kOutputShapes;
/* static */ constexpr const char* const SaveDatasetV2Op::kShardFunc;
/* static */ constexpr const char* const SaveDatasetV2Op::kShardFuncOtherArgs;
/* static */ constexpr const char* const SaveDatasetV2Op::kUseShardFunc;
/* static */ constexpr const char* const SaveDatasetV2Op::kShardFuncTarguments;
/* static */ constexpr const int SaveDatasetV2Op::kFileFormatVersion;
/* static */ constexpr const char* const LoadDatasetOp::kCompression;
/* static */ constexpr const char* const LoadDatasetOp::kDatasetType;
/* static */ constexpr const char* const LoadDatasetOp::kOutputTypes;
/* static */ constexpr const char* const LoadDatasetOp::kOutputShapes;
/* static */ constexpr const char* const LoadDatasetOp::kPath;
/* static */ constexpr const char* const LoadDatasetOp::kReaderFunc;
/* static */ constexpr const char* const LoadDatasetOp::kReaderFuncOtherArgs;
/* static */ constexpr const char* const LoadDatasetOp::kReaderFuncTarguments;

SaveDatasetOp::SaveDatasetOp(OpKernelConstruction* ctx)
    : HybridAsyncOpKernel(ctx, "tf_data_save_dataset") {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCompression, &compression_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kShardFunc, /*params=*/{},
                                               &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseShardFunc, &use_shard_func_));
}

Status SaveDatasetOp::DoCompute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));

  tstring path;
  TF_RETURN_IF_ERROR(ParseScalarArgument(ctx, kPath, &path));

  // Create a run directory.
  auto run_id = random::New64();
  auto run_dir = snapshot_util::RunDirectory(path, run_id);
  TF_RETURN_IF_ERROR(ctx->env()->RecursivelyCreateDir(run_dir));
  TF_RETURN_IF_ERROR(
      WriteMetadataFile(ctx->env(), path, run_id, dataset->output_dtypes(),
                        /*num_elements=*/0, /*finalized=*/false));

  std::unique_ptr<CapturedFunction> captured_func;
  TF_RETURN_IF_ERROR(CapturedFunction::Create(
      ctx, func_metadata_, kShardFuncOtherArgs, &captured_func));

  uint64 num_elements = 0;
  TF_RETURN_IF_ERROR(WriteData(ctx, dataset, std::move(captured_func), run_dir,
                               &num_elements));
  TF_RETURN_IF_ERROR(WriteMetadataFile(ctx->env(), path, run_id,
                                       dataset->output_dtypes(), num_elements,
                                       /*finalized=*/true));
  return OkStatus();
}

Status SaveDatasetOp::WriteData(OpKernelContext* ctx, DatasetBase* dataset,
                                std::unique_ptr<CapturedFunction> captured_func,
                                const std::string& run_dir,
                                uint64* num_elements) {
  IteratorContext::Params params(ctx);
  auto function_handle_cache =
      absl::make_unique<FunctionHandleCache>(params.flr);
  params.function_handle_cache = function_handle_cache.get();
  ResourceMgr resource_mgr;
  params.resource_mgr = &resource_mgr;
  CancellationManager cancellation_manager(ctx->cancellation_manager());
  params.cancellation_manager = &cancellation_manager;

  IteratorContext iter_ctx(std::move(params));
  std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func;
  TF_RETURN_IF_ERROR(
      captured_func->Instantiate(&iter_ctx, &instantiated_captured_func));

  DatasetBase* finalized_dataset;
  TF_RETURN_IF_ERROR(FinalizeDataset(ctx, dataset, &finalized_dataset));

  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(finalized_dataset->MakeIterator(
      &iter_ctx, /*parent=*/nullptr, "Save", &iterator));

  mutex mu;
  Status status;
  absl::flat_hash_map<int64_t, std::unique_ptr<snapshot_util::AsyncWriter>>
      writers;
  while (true) {
    if (ctx->cancellation_manager()->IsCancelled()) {
      return errors::Cancelled("Operation was cancelled");
    }
    std::vector<Tensor> element;
    bool end_of_input;
    TF_RETURN_IF_ERROR(iterator->GetNext(&iter_ctx, &element, &end_of_input));
    if (end_of_input) {
      break;
    }
    (*num_elements)++;

    // Run the shard function to compute the shard index.
    int64_t shard_index = -1;
    TF_RETURN_IF_ERROR(GetShardIndex(
        &iter_ctx, instantiated_captured_func.get(), element, &shard_index));

    // If the index does not exist, we will start a new thread.
    if (writers.count(shard_index) == 0) {
      const auto snapshot_shard_directory =
          snapshot_util::ShardDirectory(run_dir, shard_index);
      auto writer_thread = std::make_unique<snapshot_util::AsyncWriter>(
          ctx->env(), shard_index, snapshot_shard_directory,
          /*checkpoint_id=*/0, compression_, kFileFormatVersion,
          finalized_dataset->output_dtypes(), [&mu, &status](Status s) {
            mutex_lock l(mu);
            status.Update(s);
          });
      writers.insert({shard_index, std::move(writer_thread)});
    }
    writers[shard_index]->Write(element);
  }

  // Push the end of sequence signal to each of the threads to close files.
  for (auto& writer : writers) {
    writer.second->SignalEOF();
  }
  // Wait for the writer threads to join.
  writers.clear();

  return status;
}

Status SaveDatasetOp::GetShardIndex(IteratorContext* ctx,
                                    InstantiatedCapturedFunction* function,
                                    const std::vector<Tensor>& element,
                                    int64_t* shard_index) {
  if (!use_shard_func_) {
    *shard_index = (*shard_index + 1) % GetCpuBudget();
    return OkStatus();
  }
  std::vector<Tensor> output_tensors;
  TF_RETURN_IF_ERROR(function->RunWithBorrowedArgs(
      ctx, element, &output_tensors, /*node=*/nullptr));

  if (output_tensors.size() != 1 || output_tensors[0].dtype() != DT_INT64 ||
      output_tensors[0].NumElements() != 1) {
    return errors::InvalidArgument("`shard_func` must return a scalar int64.");
  }
  *shard_index = output_tensors[0].flat<int64_t>()(0);
  return OkStatus();
}

Status SaveDatasetOp::WriteMetadataFile(Env* env, const std::string& path,
                                        uint64 run_id,
                                        const DataTypeVector& output_dtypes,
                                        uint64 num_elements, bool finalized) {
  SnapshotMetadataRecord metadata;
  metadata.set_creation_timestamp(EnvTime::NowMicros());
  metadata.set_run_id(
      strings::Printf("%llu", static_cast<unsigned long long>(run_id)));
  metadata.set_version(kFileFormatVersion);
  for (const auto& output_dtype : output_dtypes) {
    metadata.add_dtype(output_dtype);
  }
  metadata.set_finalized(finalized);
  metadata.set_num_elements(num_elements);
  return snapshot_util::WriteMetadataFile(env, path, &metadata);
}

class SaveDatasetV2Op::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, const tstring& path,
          const std::string& compression,
          std::unique_ptr<CapturedFunction> shard_func, bool use_shard_func)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        path_(path),
        compression_(compression),
        shard_func_(std::move(shard_func)),
        use_shard_func_(use_shard_func) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal() const override { return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return OkStatus();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    Node* path_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(path_, &path_node));

    std::vector<Node*> shard_func_other_args;
    DataTypeVector shard_func_other_args_types;
    TF_RETURN_IF_ERROR(shard_func_->AddToGraph(ctx, b, &shard_func_other_args,
                                               &shard_func_other_args_types));

    // Attr: compression
    AttrValue compression_attr;
    b->BuildAttrValue(compression_, &compression_attr);

    // Attr: shard_func
    AttrValue shard_func_attr;
    b->BuildAttrValue(shard_func_->func(), &shard_func_attr);

    // Attr: use_shard_func
    AttrValue use_shard_func_attr;
    b->BuildAttrValue(use_shard_func_, &use_shard_func_attr);

    // Attr: shard_func_arguments_types
    AttrValue shard_func_arguments_types_attr;
    b->BuildAttrValue(shard_func_other_args_types,
                      &shard_func_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        /*inputs=*/
        {std::make_pair(0, input_graph_node), std::make_pair(1, path_node)},
        /*list_inputs=*/
        {std::make_pair(2, shard_func_other_args)},
        /*attrs=*/
        {std::make_pair(kCompression, compression_attr),
         std::make_pair(kShardFunc, shard_func_attr),
         std::make_pair(kUseShardFunc, use_shard_func_attr),
         std::make_pair(kShardFuncTarguments, shard_func_arguments_types_attr)},
        output));

    return OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    static constexpr const char* const kIteratorName = "Writer";
    static constexpr const char* const kRunId = "run_id";
    static constexpr const char* const kCurrentCheckpointId =
        "current_checkpoint_id";

    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          writers_closed_(false),
          run_id_(0),
          current_checkpoint_id_(0) {}

    ~Iterator() override {
      mutex_lock l(mu_);
      SignalEOF(true);
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(
          dataset()->shard_func_->Instantiate(ctx, &instantiated_shard_func_));

      // If we are restoring from a checkpointed iterator, we initialize
      // the run directory within the RestoreInternal method.
      if (!ctx->is_restoring()) {
        run_id_ = random::New64();
        run_dir_ = snapshot_util::RunDirectory(
            io::JoinPath(dataset()->writer_prefix_, dataset()->path_), run_id_);
        TF_RETURN_IF_ERROR(ctx->env()->RecursivelyCreateDir(run_dir_));
        TF_RETURN_IF_ERROR(WriteMetadataFile(
            ctx->env(), dataset()->path_, run_id_, dataset()->output_dtypes(),
            /*num_elements=*/0, /*finalized=*/false));
      }
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      *end_of_sequence = false;
      snapshot_util::AsyncWriter* current_writer;

      {
        std::vector<Tensor> output_tensors;
        mutex_lock l(mu_);

        // Writers have either encountered an error or are closed.
        {
          mutex_lock wsl(writer_status_mu_);
          if (!writer_status_.ok() || writers_closed_) {
            *end_of_sequence = true;
            return writer_status_;
          }
        }

        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));

        // Finalize metadata file when we are at the end of the iterator.
        if (*end_of_sequence) {
          SignalEOF(/*mark_closed=*/true);
          {
            mutex_lock wsl(writer_status_mu_);
            TF_RETURN_IF_ERROR(writer_status_);
          }
          return WriteMetadataFile(
              ctx->env(), dataset()->path_, run_id_, dataset()->output_dtypes(),
              dataset()->Cardinality(), /*finalized=*/true);
        }
        (num_elements_)++;

        int64_t shard_index = 0;
        TF_RETURN_IF_ERROR(
            GetShardIndex(ctx, instantiated_shard_func_.get(), *out_tensors,
                          dataset()->use_shard_func_, &shard_index));

        // If the index does not exist, we will start a new thread.
        if (writers_.count(shard_index) == 0) {
          auto snapshot_shard_directory =
              snapshot_util::ShardDirectory(run_dir_, shard_index);
          auto writer = std::make_unique<snapshot_util::AsyncWriter>(
              ctx->env(), shard_index, snapshot_shard_directory,
              current_checkpoint_id_, dataset()->compression_,
              kFileFormatVersion, dataset()->output_dtypes(), [this](Status s) {
                if (!s.ok()) {
                  mutex_lock l(writer_status_mu_);
                  writer_status_ = s;
                }
              });
          writers_.insert({shard_index, std::move(writer)});
        }
        current_writer = writers_[shard_index].get();
      }

      current_writer->Write(*out_tensors);
      return OkStatus();
    }

   protected:
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunId),
                                             static_cast<int64_t>(run_id_)));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCurrentCheckpointId),
                              static_cast<int64_t>(current_checkpoint_id_)));
      SignalEOF(/*mark_closed=*/false);
      writers_.clear();
      current_checkpoint_id_++;
      return SaveInput(ctx, writer, input_impl_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      int64_t run_id_signed;
      int64_t current_checkpoint_id;

      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunId), &run_id_signed));
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentCheckpointId),
                                            &current_checkpoint_id));

      run_id_ = static_cast<uint64>(run_id_signed);
      run_dir_ = snapshot_util::RunDirectory(
          io::JoinPath(dataset()->writer_prefix_, dataset()->path_), run_id_);
      current_checkpoint_id_ = static_cast<uint64>(current_checkpoint_id);

      if (ctx->is_restoring()) {
        TF_RETURN_IF_ERROR(ctx->env()->RecursivelyCreateDir(run_dir_));
        TF_RETURN_IF_ERROR(WriteMetadataFile(
            ctx->env(), dataset()->path_, run_id_, dataset()->output_dtypes(),
            /*num_elements=*/0, /*finalized=*/false));
      }

      return RestoreInput(ctx, reader, input_impl_);
    }

   private:
    Status GetShardIndex(IteratorContext* ctx,
                         InstantiatedCapturedFunction* function,
                         const std::vector<Tensor>& element,
                         bool use_shard_func, int64_t* shard_index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!use_shard_func) {
        *shard_index = (*shard_index + 1) % GetCpuBudget();
        return OkStatus();
      }
      std::vector<Tensor> output_tensors;
      TF_RETURN_IF_ERROR(function->RunWithBorrowedArgs(
          ctx, element, &output_tensors, /*node=*/nullptr));

      if (output_tensors.size() != 1 || output_tensors[0].dtype() != DT_INT64 ||
          output_tensors[0].NumElements() != 1) {
        return errors::InvalidArgument(
            "`shard_func` must return a scalar int64.");
      }
      *shard_index = output_tensors[0].flat<int64_t>()(0);
      return OkStatus();
    }

    Status WriteMetadataFile(Env* env, const std::string& path, uint64 run_id,
                             const DataTypeVector& output_dtypes,
                             uint64 num_elements, bool finalized)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      SnapshotMetadataRecord metadata;
      metadata.set_creation_timestamp(EnvTime::NowMicros());
      metadata.set_run_id(
          strings::Printf("%llu", static_cast<unsigned long long>(run_id)));
      metadata.set_version(kFileFormatVersion);
      for (const auto& output_dtype : output_dtypes) {
        metadata.add_dtype(output_dtype);
      }
      metadata.set_finalized(finalized);
      metadata.set_num_elements(num_elements);
      return snapshot_util::WriteMetadataFile(env, path, &metadata);
    }

    void SignalEOF(bool mark_closed) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!writers_closed_) {
        for (auto& writer : writers_) {
          writer.second->SignalEOF();
        }

        writers_.clear();
        writers_closed_ = mark_closed;
      }
    }

    mutex mu_;
    mutex writer_status_mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    int64_t num_elements_;

    absl::flat_hash_map<int64_t, std::unique_ptr<snapshot_util::AsyncWriter>>
        writers_ TF_GUARDED_BY(mu_);
    Status writer_status_ TF_GUARDED_BY(writer_status_mu_);
    bool writers_closed_ TF_GUARDED_BY(mu_);

    uint64 run_id_ TF_GUARDED_BY(mu_);
    tstring run_dir_ TF_GUARDED_BY(mu_);

    uint64 current_checkpoint_id_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_shard_func_
        TF_GUARDED_BY(mu_);
  };

  const DatasetBase* input_;
  const tstring path_;
  const std::string compression_;
  const std::unique_ptr<CapturedFunction> shard_func_;
  const bool use_shard_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::shared_ptr<FunctionMetadata> func_metadata_;
  const std::string writer_prefix_;
};

SaveDatasetV2Op::SaveDatasetV2Op(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCompression, &compression_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseShardFunc, &use_shard_func_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kShardFunc, /*params=*/{},
                                               &func_metadata_));
}

void SaveDatasetV2Op::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
  DatasetBase* dataset;
  OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

  tstring path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, kPath, &path));

  std::unique_ptr<CapturedFunction> shard_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(ctx, func_metadata_, kShardFuncOtherArgs,
                                    &shard_func));

  *output = new Dataset(ctx, dataset, path, compression_, std::move(shard_func),
                        use_shard_func_);
}

class LoadDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const tstring& path,
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
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal() const override {
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
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

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
      auto run_dir = snapshot_util::RunDirectory(dataset()->path_,
                                                 dataset()->metadata_.run_id());

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

  bool metadata_file_exists;
  experimental::SnapshotMetadataRecord metadata;
  OP_REQUIRES_OK(ctx, snapshot_util::ReadMetadataFile(
                          ctx->env(), path, &metadata, &metadata_file_exists));

  OP_REQUIRES(ctx, metadata_file_exists,
              errors::NotFound("Could not find metadata file [", path, "]"));

  *output =
      new Dataset(ctx, path, std::move(metadata), compression_,
                  std::move(captured_func), output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("SaveDataset").Device(DEVICE_CPU), SaveDatasetOp);
REGISTER_KERNEL_BUILDER(Name("SaveDatasetV2").Device(DEVICE_CPU),
                        SaveDatasetV2Op);
REGISTER_KERNEL_BUILDER(Name("LoadDataset").Device(DEVICE_CPU), LoadDatasetOp);
}  // namespace

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
