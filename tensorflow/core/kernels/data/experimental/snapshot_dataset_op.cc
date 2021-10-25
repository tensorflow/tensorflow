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
#include "tensorflow/core/kernels/data/experimental/snapshot_dataset_op.h"

#include <random>

#include "absl/time/clock.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/hash_utils.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/raw_coding.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/snappy.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"
#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#endif  // IS_SLIM_BUILD
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/base64.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/core/util/batch_util.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const SnapshotDatasetV2Op::kDatasetType;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kOutputTypes;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kOutputShapes;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kCompression;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kReaderPrefix;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kWriterPrefix;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kHashValid;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kHash;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kCompressionAuto;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kReaderFunc;
/* static */ constexpr const char* const SnapshotDatasetV2Op::kShardFunc;
/* static */ constexpr const char* const
    SnapshotDatasetV2Op::kReaderFuncOtherArgs;
/* static */ constexpr const char* const
    SnapshotDatasetV2Op::kShardFuncOtherArgs;
/* static */ constexpr const char* const
    SnapshotDatasetV2Op::kReaderFuncTarguments;
/* static */ constexpr const char* const
    SnapshotDatasetV2Op::kShardFuncTarguments;
/* static */ constexpr const int SnapshotDatasetV2Op::kFileFormatVersion;

// ==== Snapshot Implementation ====

/* The current snapshot on-disk layout is as follows:
 *   /user/specified/path/
 *     - graphhash1/
 *       - snapshot.metadata  // metadata file
 *       - run1/
 *         - 00000000.shard/  // shard index
 *           // new checkpoint files are created on all threads at once, either
 *           // when a file gets too big, or when a TF checkpoint happens.
 *           - 00000000.snapshot  // checkpoint file 0
 *           - 00000001.snapshot  // checkpoint file 1
 *           - ...
 *         - 00000001.shard/
 *           - 00000000.snapshot
 *           - 00000001.snapshot
 *           - ...
 *         - 00000002.shard/
 *           - 00000000.snapshot
 *           - 00000001.snapshot
 *           - ...
 *           ...
 *       - run2/
 *           ...
 *     - graphhash2/
 *       ...
 *     - graphhash3/
 *       ...
 */

class SnapshotDatasetV2Op::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, uint64 hash,
          const std::string& path, const std::string& compression,
          const std::string& reader_prefix, const std::string& writer_prefix,
          std::unique_ptr<CapturedFunction> reader_func,
          std::unique_ptr<CapturedFunction> shard_func)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        hash_(hash),
        path_(path),
        compression_(compression),
        reader_prefix_(reader_prefix),
        writer_prefix_(writer_prefix),
        reader_func_(std::move(reader_func)),
        shard_func_(std::move(shard_func)) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, absl::StrCat(prefix, "::Snapshot")});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
    return errors::Unimplemented(
        "Splitting is not implemented for snapshot datasets.");
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
    return Status::OK();
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

    Node* path = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(path_, &path));

    std::vector<Node*> reader_func_other_args;
    DataTypeVector reader_func_other_args_types;
    TF_RETURN_IF_ERROR(reader_func_->AddToGraph(ctx, b, &reader_func_other_args,
                                                &reader_func_other_args_types));

    std::vector<Node*> shard_func_other_args;
    DataTypeVector shard_func_other_args_types;
    TF_RETURN_IF_ERROR(shard_func_->AddToGraph(ctx, b, &shard_func_other_args,
                                               &shard_func_other_args_types));

    AttrValue compression_attr;
    b->BuildAttrValue(compression_, &compression_attr);

    AttrValue reader_prefix_attr;
    b->BuildAttrValue(reader_prefix_, &reader_prefix_attr);

    AttrValue writer_prefix_attr;
    b->BuildAttrValue(writer_prefix_, &writer_prefix_attr);

    AttrValue hash_valid_attr;
    b->BuildAttrValue(true, &hash_valid_attr);

    AttrValue hash_attr;
    b->BuildAttrValue(static_cast<int64_t>(hash_), &hash_attr);

    AttrValue reader_func_attr;
    b->BuildAttrValue(reader_func_->func(), &reader_func_attr);

    AttrValue shard_func_attr;
    b->BuildAttrValue(shard_func_->func(), &shard_func_attr);

    AttrValue reader_func_arguments_types_attr;
    b->BuildAttrValue(reader_func_other_args_types,
                      &reader_func_arguments_types_attr);

    AttrValue shard_func_arguments_types_attr;
    b->BuildAttrValue(shard_func_other_args_types,
                      &shard_func_arguments_types_attr);

    return b->AddDataset(
        this,
        /*inputs=*/
        {std::make_pair(0, input_graph_node), std::make_pair(1, path)},
        /*list_inputs=*/
        {std::make_pair(2, reader_func_other_args),
         std::make_pair(3, shard_func_other_args)},
        /*attrs=*/
        {{kCompression, compression_attr},
         {kReaderPrefix, reader_prefix_attr},
         {kWriterPrefix, writer_prefix_attr},
         {kHashValid, hash_valid_attr},
         {kHash, hash_attr},
         {kReaderFunc, reader_func_attr},
         {kShardFunc, shard_func_attr},
         {kReaderFuncTarguments, reader_func_arguments_types_attr},
         {kShardFuncTarguments, shard_func_arguments_types_attr}},
        output);
  }

 private:
  const DatasetBase* input_;
  const uint64 hash_;
  const tstring path_;
  const std::string compression_;
  const std::string reader_prefix_;
  const std::string writer_prefix_;

  std::unique_ptr<CapturedFunction> reader_func_;
  std::unique_ptr<CapturedFunction> shard_func_;

  class Reader : public DatasetIterator<Dataset> {
   public:
    static constexpr const char* const kIteratorName = "Reader";

    Reader(const Params& params, int64_t start_index)
        : DatasetIterator<Dataset>(params), start_index_(start_index) {}

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);

      TF_RETURN_IF_ERROR(dataset()->reader_func_->Instantiate(
          ctx, &instantiated_reader_func_));

      auto hash_dir = snapshot_util::HashDirectory(
          io::JoinPath(dataset()->reader_prefix_, dataset()->path_),
          dataset()->hash_);
      bool metadata_file_exists;
      experimental::SnapshotMetadataRecord metadata;
      TF_RETURN_IF_ERROR(snapshot_util::ReadMetadataFile(
          ctx->env(), hash_dir, &metadata, &metadata_file_exists));

      auto run_dir = snapshot_util::RunDirectory(hash_dir, metadata.run_id());

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
          metadata.version(), dataset()->output_dtypes(),
          dataset()->output_shapes(), start_index_,
          &dataset_of_snapshot_files));

      Tensor input_dataset_tensor(DT_VARIANT, TensorShape({}));
      TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(dataset_of_snapshot_files,
                                                     &input_dataset_tensor));

      std::vector<Tensor> reader_input;
      std::vector<Tensor> reader_output;
      reader_input.push_back(std::move(input_dataset_tensor));

      // NOTE: We intentionally ignore resource modeling outside GetNext().
      TF_RETURN_IF_ERROR(instantiated_reader_func_->Run(
          ctx, std::move(reader_input), &reader_output, /*node=*/nullptr));
      if (reader_output.size() != 1) {
        return errors::InvalidArgument(
            "reader_func returns more than one argument.");
      }
      TF_RETURN_IF_ERROR(
          GetDatasetFromVariantTensor(reader_output[0], &input_));
      return input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      // We do not need to checkpoint the reader as we are rebuilding the
      // reader datasets from information that is already saved by the main
      // iterator.
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return Status::OK();
    }

   private:
    const int64_t start_index_;

    mutex mu_;

    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);

    DatasetBase* input_ TF_GUARDED_BY(mu_) = nullptr;

    std::unique_ptr<InstantiatedCapturedFunction> instantiated_reader_func_
        TF_GUARDED_BY(mu_);
  };

  class Writer : public DatasetIterator<Dataset> {
   public:
    static constexpr const char* const kIteratorName = "Writer";
    static constexpr const char* const kRunId = "run_id";
    static constexpr const char* const kCurrentCheckpointId =
        "current_checkpoint_id";

    explicit Writer(const Params& params)
        : DatasetIterator<Dataset>(params),
          writers_closed_(false),
          run_id_(0),
          current_checkpoint_id_(0) {}

    ~Writer() override {
      mutex_lock l(mu_);
      SignalEOF(true);
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(
          dataset()->shard_func_->Instantiate(ctx, &instantiated_shard_func_));

      return dataset()->input_->MakeIterator(
          ctx, this, strings::StrCat(prefix(), "::WriterIterator"),
          &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      *end_of_sequence = false;
      snapshot_util::AsyncWriter* current_writer;

      {
        std::vector<Tensor> output_tensors;
        mutex_lock l(mu_);

        // We initialize late here because restoring from checkpoint comes
        // after the the Initialize call. We cannot initialize within
        // Initialize() because we cannot determine whether we should
        // overwrite an existing metadata file or not before `RestoreInternal`
        // is potentially called.
        if (run_dir_.empty()) {
          run_id_ = random::New64();

          // Creates the run directory.
          run_dir_ = snapshot_util::RunDirectory(
              snapshot_util::HashDirectory(
                  io::JoinPath(dataset()->writer_prefix_, dataset()->path_),
                  dataset()->hash_),
              run_id_);
          TF_RETURN_IF_ERROR(ctx->env()->RecursivelyCreateDir(run_dir_));
          TF_RETURN_IF_ERROR(
              WriteMetadataFile(ctx->env(), /*finalized=*/false));
        }

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
          return WriteMetadataFile(ctx->env(), /*finalized=*/true);
        }

        int64_t shard_index = 0;
        TF_RETURN_IF_ERROR(GetShardIndex(ctx, *out_tensors, &shard_index));

        // If the index does not exist, we will start a new thread.
        if (writers_.count(shard_index) == 0) {
          auto snapshot_shard_directory =
              snapshot_util::ShardDirectory(run_dir_, shard_index);
          auto writer = std::make_unique<snapshot_util::AsyncWriter>(
              ctx->env(), shard_index, snapshot_shard_directory,
              current_checkpoint_id_, dataset()->compression_,
              kFileFormatVersion, dataset()->output_dtypes(), [this](Status s) {
                if (!s.ok()) {
                  LOG(ERROR) << "AsyncWriter in snapshot writer failed: " << s;
                  mutex_lock l(writer_status_mu_);
                  writer_status_ = s;
                }
              });
          writers_.insert({shard_index, std::move(writer)});
        }
        current_writer = writers_[shard_index].get();
      }

      current_writer->Write(*out_tensors);
      return Status::OK();
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
          snapshot_util::HashDirectory(
              io::JoinPath(dataset()->writer_prefix_, dataset()->path_),
              dataset()->hash_),
          run_id_);
      current_checkpoint_id_ = static_cast<uint64>(current_checkpoint_id);

      return RestoreInput(ctx, reader, input_impl_);
    }

   private:
    Status GetShardIndex(IteratorContext* ctx,
                         const std::vector<Tensor>& tensors,
                         int64_t* shard_index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      std::vector<Tensor> output_tensors;

      // Run the shard function
      TF_RETURN_IF_ERROR(instantiated_shard_func_->RunWithBorrowedArgs(
          ctx, tensors, &output_tensors, model_node()));

      if (output_tensors.size() != 1 || output_tensors[0].dtype() != DT_INT64 ||
          output_tensors[0].NumElements() != 1) {
        return errors::InvalidArgument(
            "`shard_func` must return a scalar int64.");
      }

      // Create writable files if we see an index bigger than our current
      // files.
      *shard_index = output_tensors[0].flat<int64_t>()(0);
      return Status::OK();
    }

    Status WriteMetadataFile(Env* env, bool finalized)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      DCHECK(!run_dir_.empty());

      experimental::SnapshotMetadataRecord metadata;
      metadata.set_creation_timestamp(EnvTime::NowMicros());
      metadata.set_graph_hash(strings::StrCat(dataset()->hash_));
      metadata.set_run_id(strings::StrCat(run_id_));
      metadata.set_version(kFileFormatVersion);
      for (const auto& output_dtype : dataset()->output_dtypes()) {
        metadata.add_dtype(output_dtype);
      }
      metadata.set_finalized(finalized);
      tstring hash_directory = io::JoinPath(
          dataset()->writer_prefix_,
          snapshot_util::HashDirectory(dataset()->path_, dataset()->hash_));

      return snapshot_util::WriteMetadataFile(env, hash_directory, &metadata);
    }

    void SignalEOF(bool mark_closed) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!writers_closed_) {
        // Push the end of sequence signal to each of the threads to close
        // files.
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

    absl::flat_hash_map<int64_t, std::unique_ptr<snapshot_util::AsyncWriter>>
        writers_ TF_GUARDED_BY(mu_);
    Status writer_status_ TF_GUARDED_BY(writer_status_mu_);
    bool writers_closed_ TF_GUARDED_BY(mu_);

    uint64 run_id_ TF_GUARDED_BY(mu_);
    tstring run_dir_ TF_GUARDED_BY(mu_);

    // Stores the ID of the current checkpoint .snapshot file being read. See
    // top of this file for the directory layout.
    uint64 current_checkpoint_id_ TF_GUARDED_BY(mu_);

    std::unique_ptr<InstantiatedCapturedFunction> instantiated_shard_func_
        TF_GUARDED_BY(mu_);
  };

  class Passthrough : public DatasetIterator<Dataset> {
   public:
    static constexpr const char* const kIteratorName = "Passthrough";

    explicit Passthrough(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      return SaveInput(ctx, writer, input_impl_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return RestoreInput(ctx, reader, input_impl_);
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  class Iterator : public DatasetIterator<Dataset> {
   public:
    static constexpr const char* const kIteratorMode = "iterator_mode";
    static constexpr const char* const kIndex = "index";
    static constexpr const char* const kGraphHashDirectory =
        "graph_hash_directory";

    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          index_(0),
          hash_dir_(snapshot_util::HashDirectory(dataset()->path_,
                                                 dataset()->hash_)) {}

    Status Initialize(IteratorContext* ctx) override {
      return ctx->env()->RecursivelyCreateDir(
          io::JoinPath(dataset()->writer_prefix_, hash_dir_));
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (iterator_ == nullptr) {
        Status s = InitializeIterator(ctx, /*reader=*/nullptr);
        if (!s.ok()) {
          iterator_.reset();
          return s;
        }
      }
      index_++;
      return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (iterator_ != nullptr) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, iterator_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kIteratorMode),
                                               static_cast<int64_t>(mode_)));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kIndex), index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kGraphHashDirectory), hash_dir_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      if (reader->Contains(full_name(kIteratorMode))) {
        TF_RETURN_IF_ERROR(InitializeIterator(ctx, reader));
        return RestoreInput(ctx, reader, iterator_);
      }
      return Status::OK();
    }

   private:
    Status InitializeIterator(IteratorContext* ctx, IteratorStateReader* reader)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (reader != nullptr) {
        // Check whether the computed hash directory is the same.
        tstring hash_dir;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(kGraphHashDirectory), &hash_dir));
        if (hash_dir != hash_dir_) {
          return errors::DataLoss(
              "Dataset has changed while restoring from the checkpoint. Old "
              "hash "
              "directory: ",
              hash_dir, "; new hash directory: ", hash_dir_);
        }

        experimental::SnapshotMetadataRecord metadata;
        bool file_exists;
        TF_RETURN_IF_ERROR(snapshot_util::ReadMetadataFile(
            ctx->env(), io::JoinPath(dataset()->reader_prefix_, hash_dir_),
            &metadata, &file_exists));
        if (!file_exists) {
          return errors::DataLoss("Snapshot metadata file in ", hash_dir_,
                                  " does not exist any more.");
        }

        int64_t iterator_mode;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(kIteratorMode), &iterator_mode));
        mode_ = snapshot_util::Mode(iterator_mode);

        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kIndex), &index_));
      } else {
        experimental::SnapshotMetadataRecord metadata;
        bool file_exists;
        TF_RETURN_IF_ERROR(snapshot_util::ReadMetadataFile(
            ctx->env(), io::JoinPath(dataset()->reader_prefix_, hash_dir_),
            &metadata, &file_exists));

        // `pending_snapshot_expiry_seconds` is a legacy option where we would
        // not write snapshots that we think were still on-going. We decided
        // that this would not be necessary as a feature for SnapshotV2, and we
        // would always write a new snapshot regardless of whether someone else
        // is currently writing one. Setting this to 0 ensures that all previous
        // snapshots will be ignored and we will proceed to writing.
        TF_RETURN_IF_ERROR(snapshot_util::DetermineOpState(
            /*mode_string=*/"", file_exists, &metadata,
            /*pending_snapshot_expiry_seconds=*/0, &mode_));
      }

      switch (mode_) {
        case snapshot_util::READER:
          iterator_ = absl::make_unique<Reader>(
              Reader::Params{dataset(),
                             absl::StrCat(prefix(), Reader::kIteratorName)},
              index_);
          break;
        case snapshot_util::WRITER:
          iterator_ = absl::make_unique<Writer>(Writer::Params{
              dataset(), absl::StrCat(prefix(), Writer::kIteratorName)});
          break;
        case snapshot_util::PASSTHROUGH:
          iterator_ = absl::make_unique<Passthrough>(Passthrough::Params{
              dataset(), absl::StrCat(prefix(), Passthrough::kIteratorName)});
          break;
      }
      TF_RETURN_IF_ERROR(iterator_->InitializeBase(ctx, this));
      return iterator_->Initialize(ctx);
    }

    mutex mu_;
    int64_t index_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> iterator_ TF_GUARDED_BY(mu_);
    snapshot_util::Mode mode_ TF_GUARDED_BY(mu_);
    const std::string hash_dir_;
  };
};

SnapshotDatasetV2Op::SnapshotDatasetV2Op(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
  FunctionMetadata::Params reader_params;
  FunctionMetadata::Params shard_params;

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCompression, &compression_));

  if (ctx->HasAttr(kReaderPrefix)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kReaderPrefix, &reader_prefix_));
  }

  if (ctx->HasAttr(kWriterPrefix)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kWriterPrefix, &writer_prefix_));
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kHashValid, &hash_valid_));
  int64_t hash;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kHash, &hash));
  hash_ = static_cast<uint64>(hash);

  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kReaderFunc, reader_params,
                                               &reader_func_metadata_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kShardFunc, shard_params,
                                               &shard_func_metadata_));
}

void SnapshotDatasetV2Op::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                      DatasetBase** output) {
  tstring path;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "path", &path));

  std::string compression = compression_ == kCompressionAuto
                                ? io::compression::kSnappy
                                : compression_;
  uint64 hash;
  if (hash_valid_) {
    hash = hash_;
  } else {
    // Computes the hash of the preceding items in the graph.
    GraphDef graph_def;
    SerializationContext::Params params(ctx);
    std::vector<std::pair<string, Tensor>> input_list;
    params.input_list = &input_list;
    params.external_state_policy =
        SerializationContext::ExternalStatePolicy::kIgnore;
    OP_REQUIRES_OK(
        ctx, AsGraphDef(ctx, input, SerializationContext(params), &graph_def));
    OP_REQUIRES_OK(ctx, HashGraph(graph_def, &hash));
    // Different compression modes should result in different graph hashes.
    hash = Hash64Combine(hash, Hash64(compression));
  }

  std::unique_ptr<CapturedFunction> reader_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, reader_func_metadata_,
                                          kReaderFuncOtherArgs, &reader_func));
  std::unique_ptr<CapturedFunction> shard_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, shard_func_metadata_,
                                          kShardFuncOtherArgs, &shard_func));

  *output = new SnapshotDatasetV2Op::Dataset(
      ctx, input, hash, path, compression, reader_prefix_, writer_prefix_,
      std::move(reader_func), std::move(shard_func));
}

namespace {
REGISTER_KERNEL_BUILDER(Name("SnapshotDatasetV2").Device(DEVICE_CPU),
                        SnapshotDatasetV2Op);
}  // namespace

// ==== Legacy Snapshot Implementation (Deprecated) ====

namespace {

// Defaults to 10 GiB per shard.
const int64_t kDefaultShardSizeBytes = 10LL * 1024 * 1024 * 1024;

const int64_t kCurrentVersion = 1;

constexpr char kSnapshotReaderWorkerPool[] = "snapshot_reader_worker_pool";
constexpr char kSnapshotWriterWorkerPool[] = "snapshot_writer_worker_pool";
constexpr char kSeparator[] = "::";
constexpr char kBookkeeping[] = "Bookkeeping";
constexpr char kSnapshotReadElements[] = "snapshot_read_elements";
constexpr char kSnapshotReadThroughput[] = "snapshot_read_throughput";
constexpr char kSnapshotWrittenElements[] = "snapshot_written_elements";
constexpr char kSnapshotWriteThroughput[] = "snapshot_write_throughput";

constexpr char kSizeSuffix[] = "_size";
constexpr char kState[] = "state";
constexpr char kHashDir[] = "hash_dir";
constexpr char kRunId[] = "run_id";
constexpr char kRunDir[] = "run_dir";
constexpr char kVersionStr[] = "version";
constexpr char kFilenames[] = "filenames";
constexpr char kCurrentFilenames[] = "current_filenames";
constexpr char kElementsProduced[] = "elements_produced";
constexpr char kNextFileIndex[] = "next_file_index";
constexpr char kNumFilesDone[] = "num_files_done";
constexpr char kNumElementsRead[] = "num_elements_read";
constexpr char kStatus[] = "status";
constexpr char kCode[] = ".code";
constexpr char kErrorMessage[] = ".error_message";
constexpr char kEndOfSequence[] = "end_of_sequence";
constexpr char kBuffer[] = "buffer";
constexpr char kNumElementsWritten[] = "num_elements_written";
constexpr char kNextElem[] = "next_elem";

class SnapshotDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SnapshotDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));

    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("reader_path_prefix", &reader_path_prefix_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("writer_path_prefix", &writer_path_prefix_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("compression", &compression_));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("shard_size_bytes", &shard_size_bytes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pending_snapshot_expiry_seconds",
                                     &pending_snapshot_expiry_seconds_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("num_reader_threads", &num_reader_threads_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("reader_buffer_size", &reader_buffer_size_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("num_writer_threads", &num_writer_threads_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("writer_buffer_size", &writer_buffer_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shuffle_on_read", &shuffle_on_read_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2_));

    mode_ = snapshot_util::kModeAuto;
    if (ctx->HasAttr("mode")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_));
    }

    snapshot_name_ = "";
    if (ctx->HasAttr("snapshot_name")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("snapshot_name", &snapshot_name_));
    }

    if (shard_size_bytes_ == -1) shard_size_bytes_ = kDefaultShardSizeBytes;

    // Default to 1 day expiry for snapshots.
    if (pending_snapshot_expiry_seconds_ == -1) {
      pending_snapshot_expiry_seconds_ = 86400;
    }

    if (num_reader_threads_ == -1) num_reader_threads_ = 1;
    if (reader_buffer_size_ == -1) reader_buffer_size_ = 1;
    if (num_writer_threads_ == -1) num_writer_threads_ = 1;
    if (writer_buffer_size_ == -1) writer_buffer_size_ = 1;

    OP_REQUIRES(
        ctx,
        compression_ == io::compression::kNone ||
            compression_ == io::compression::kGzip ||
            compression_ == io::compression::kSnappy,
        errors::InvalidArgument("compression must be either '', 'GZIP' or "
                                "'SNAPPY'."));

    OP_REQUIRES(
        ctx, pending_snapshot_expiry_seconds_ >= 1,
        errors::InvalidArgument(
            "pending_snapshot_expiry_seconds must be at least 1 second."));

    OP_REQUIRES(ctx,
                mode_ == snapshot_util::kModeAuto ||
                    mode_ == snapshot_util::kModeRead ||
                    mode_ == snapshot_util::kModeWrite ||
                    mode_ == snapshot_util::kModePassthrough,
                errors::InvalidArgument(
                    "mode must be either '", snapshot_util::kModeAuto, "', '",
                    snapshot_util::kModeRead, "', '", snapshot_util::kModeWrite,
                    "', or '", snapshot_util::kModePassthrough, "'."));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    tstring path;

    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "path", &path));

    SerializationContext::Params params(ctx);
    std::vector<std::pair<string, Tensor>> input_list;
    params.input_list = &input_list;
    params.external_state_policy =
        SerializationContext::ExternalStatePolicy::kIgnore;

    GraphDef graph_def;
    OP_REQUIRES_OK(
        ctx, AsGraphDef(ctx, input, SerializationContext(params), &graph_def));

    uint64 hash;
    OP_REQUIRES_OK(ctx, ComputeDatasetHash(graph_def, path, &hash));

    Status dump_status =
        snapshot_util::DumpDatasetGraph(ctx->env(), path, hash, &graph_def);
    if (!dump_status.ok()) {
      LOG(WARNING) << "Unable to write graphdef to disk, error: "
                   << dump_status.ToString();
    }

    std::string graph_hash =
        strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
    LOG(INFO) << "Graph def serialized to hash: " << graph_hash;

    *output = new Dataset(ctx, input, path, graph_hash, reader_path_prefix_,
                          writer_path_prefix_, compression_, shard_size_bytes_,
                          pending_snapshot_expiry_seconds_, num_reader_threads_,
                          reader_buffer_size_, num_writer_threads_,
                          writer_buffer_size_, shuffle_on_read_, seed_, seed2_,
                          mode_, snapshot_name_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input, const string& path,
            const string& graph_hash, const string& reader_path_prefix,
            const string& writer_path_prefix, const string& compression,
            const uint64 shard_size_bytes,
            const uint64 pending_snapshot_expiry_seconds,
            const uint64 num_reader_threads, const uint64 reader_buffer_size,
            const uint64 num_writer_threads, const uint64 writer_buffer_size,
            const bool shuffle_on_read, const uint64 seed, const uint64 seed2,
            const std::string& mode, const std::string& snapshot_name)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          dir_(path),
          graph_hash_(graph_hash),
          reader_path_prefix_(reader_path_prefix),
          writer_path_prefix_(writer_path_prefix),
          compression_(compression),
          shard_size_bytes_(shard_size_bytes),
          pending_snapshot_expiry_seconds_(pending_snapshot_expiry_seconds),
          num_reader_threads_(num_reader_threads),
          reader_buffer_size_(reader_buffer_size),
          num_writer_threads_(num_writer_threads),
          writer_buffer_size_(writer_buffer_size),
          shuffle_on_read_(shuffle_on_read),
          seed_(seed),
          seed2_(seed2),
          mode_(mode),
          snapshot_name_(snapshot_name) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, absl::StrCat(prefix, "::Snapshot")});
    }

    Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                  split_providers) const override {
      return errors::Unimplemented(
          "Splitting is not implemented for snapshot datasets.");
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() const override { return "SnapshotDatasetOp::Dataset"; }

    int64_t CardinalityInternal() const override {
      return input_->Cardinality();
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      inputs->push_back(input_);
      return Status::OK();
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

      Node* path = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(dir_, &path));

      AttrValue compression_attr;
      b->BuildAttrValue(compression_, &compression_attr);

      AttrValue reader_path_prefix_attr;
      b->BuildAttrValue(reader_path_prefix_, &reader_path_prefix_attr);

      AttrValue writer_path_prefix_attr;
      b->BuildAttrValue(writer_path_prefix_, &writer_path_prefix_attr);

      AttrValue shard_size_bytes_attr;
      b->BuildAttrValue<int64_t>(shard_size_bytes_, &shard_size_bytes_attr);

      AttrValue pending_snapshot_expiry_seconds_attr;
      b->BuildAttrValue<int64_t>(pending_snapshot_expiry_seconds_,
                                 &pending_snapshot_expiry_seconds_attr);

      AttrValue num_reader_threads_attr;
      b->BuildAttrValue<int64_t>(num_reader_threads_, &num_reader_threads_attr);

      AttrValue reader_buffer_size_attr;
      b->BuildAttrValue<int64_t>(reader_buffer_size_, &reader_buffer_size_attr);

      AttrValue num_writer_threads_attr;
      b->BuildAttrValue<int64_t>(num_writer_threads_, &num_writer_threads_attr);

      AttrValue writer_buffer_size_attr;
      b->BuildAttrValue<int64_t>(writer_buffer_size_, &writer_buffer_size_attr);

      AttrValue shuffle_on_read_attr;
      b->BuildAttrValue<bool>(shuffle_on_read_, &shuffle_on_read_attr);

      AttrValue seed_attr;
      b->BuildAttrValue<int64_t>(seed_, &seed_attr);

      AttrValue seed2_attr;
      b->BuildAttrValue<int64_t>(seed2_, &seed2_attr);

      AttrValue mode_attr;
      b->BuildAttrValue(mode_, &mode_attr);

      AttrValue snapshot_name_attr;
      b->BuildAttrValue(snapshot_name_, &snapshot_name_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this,
          /*inputs=*/
          {std::make_pair(0, input_graph_node), std::make_pair(1, path)},
          /*list_inputs=*/
          {},
          /*attrs=*/
          {{"compression", compression_attr},
           {"reader_path_prefix", reader_path_prefix_attr},
           {"writer_path_prefix", writer_path_prefix_attr},
           {"shard_size_bytes", shard_size_bytes_attr},
           {"pending_snapshot_expiry_seconds",
            pending_snapshot_expiry_seconds_attr},
           {"num_reader_threads", num_reader_threads_attr},
           {"reader_buffer_size", reader_buffer_size_attr},
           {"num_writer_threads", num_writer_threads_attr},
           {"writer_buffer_size", writer_buffer_size_attr},
           {"shuffle_on_read", shuffle_on_read_attr},
           {"seed", seed_attr},
           {"seed2", seed2_attr},
           {"mode", mode_attr},
           {"snapshot_name", snapshot_name_attr}},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
        if (dataset()->snapshot_name_.empty()) {
          hash_dir_ = io::JoinPath(dataset()->dir_, dataset()->graph_hash_);
        } else {
          hash_dir_ = io::JoinPath(
              dataset()->dir_,
              strings::StrCat("custom-", dataset()->snapshot_name_));
        }
      }

      // We have a somewhat non traditional pattern for iterator initialization
      // for Snapshot. The protocol is that we initialize the Reader / Writer
      // iterator on the first GetNext call. We also invoke the same
      // initialization code when restoring as well. The reason why we don't do
      // this during the Initialize call is because during Restore we call
      // Initialize at first and at that point we don't know which iterator
      // (Reader / Writer / Passthrough) we need to restore as this info is part
      // of the checkpoint.
      Status Initialize(IteratorContext* ctx) override { return Status::OK(); }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (iterator_ == nullptr) {
          experimental::SnapshotMetadataRecord metadata;
          bool file_exists;
          TF_RETURN_IF_ERROR(snapshot_util::ReadMetadataFile(
              ctx->env(), hash_dir_, &metadata, &file_exists));
          TF_RETURN_IF_ERROR(snapshot_util::DetermineOpState(
              dataset()->mode_, file_exists, &metadata,
              dataset()->pending_snapshot_expiry_seconds_, &state_));
          VLOG(2) << "Snapshot state: " << state_;
          TF_RETURN_IF_ERROR(InitializeIterator(ctx, metadata));
        }
        return iterator_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (iterator_ != nullptr) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, iterator_));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kState),
                                               static_cast<int64_t>(state_)));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kHashDir), hash_dir_));
        VLOG(2) << "Saving Snapshot iterator: " << state_;
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        tstring hash_dir;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kHashDir), &hash_dir));
        if (hash_dir != hash_dir_) {
          LOG(ERROR) << "Dataset has changed while restoring from the "
                        "checkpoint. Old hash: "
                     << hash_dir << "; new hash: " << hash_dir_;
          return Status::OK();
        }
        int64_t temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kState), &temp));
        state_ = snapshot_util::Mode(temp);
        experimental::SnapshotMetadataRecord metadata;
        bool file_exists;
        TF_RETURN_IF_ERROR(snapshot_util::ReadMetadataFile(
            ctx->env(), hash_dir_, &metadata, &file_exists));
        TF_RETURN_IF_ERROR(InitializeIterator(ctx, metadata));
        VLOG(2) << "Restoring Snapshot iterator: " << state_;
        return RestoreInput(ctx, reader, iterator_);
      }

      // This method expects that state_ is populated and it will create the
      // correct Reader / Writer / Passthrough iterator and initialize it.
      Status InitializeIterator(
          IteratorContext* ctx,
          const experimental::SnapshotMetadataRecord& metadata)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::string run_id = "";
        if (!dataset()->snapshot_name_.empty()) {
          // We have overridden the snapshot with a custom name, so we don't
          // generate random run ids, but just use the same one.
          run_id = "custom";
        }

        switch (state_) {
          case snapshot_util::WRITER:
            iterator_ = absl::make_unique<SnapshotWriterIterator>(
                SnapshotWriterIterator::Params{
                    dataset(), absl::StrCat(prefix(), "WriterImpl")},
                hash_dir_, run_id);
            break;
          case snapshot_util::READER:
            if (run_id.empty() && metadata.run_id().empty()) {
              return errors::NotFound(
                  "Could not find a valid snapshot to read.");
            }
            if (run_id.empty()) {
              run_id = metadata.run_id();
            }
            // dtypes in metadata should be the same as dataset()->output_dtypes
            if (metadata.dtype_size() != dataset()->output_dtypes().size()) {
              return errors::Internal(
                  "Expected number of dtypes: ",
                  dataset()->output_dtypes().size(),
                  " but number in snapshot: ", metadata.dtype_size());
            }
            for (int i = 0; i < metadata.dtype_size(); ++i) {
              if (metadata.dtype(i) != dataset()->output_dtypes()[i]) {
                return errors::Internal(
                    "Type: ", i,
                    " doesn't match. Snapshot: ", metadata.dtype(i),
                    "; dataset: ", dataset()->output_dtypes()[i]);
              }
            }
            iterator_ = absl::make_unique<SnapshotReaderIterator>(
                SnapshotReaderIterator::Params{
                    dataset(), absl::StrCat(prefix(), "ReaderImpl")},
                hash_dir_, run_id, metadata.version());
            break;
          case snapshot_util::PASSTHROUGH:
            iterator_ = absl::make_unique<SnapshotPassthroughIterator>(
                SnapshotPassthroughIterator::Params{
                    dataset(), absl::StrCat(prefix(), "PassthroughImpl")});
            break;
        }
        TF_RETURN_IF_ERROR(iterator_->InitializeBase(ctx, this));
        return iterator_->Initialize(ctx);
      }

     protected:
      class SnapshotReaderIterator : public DatasetIterator<Dataset> {
       public:
        static constexpr const char* const kParse = "Parse";

        explicit SnapshotReaderIterator(const Params& params,
                                        const string& hash_dir,
                                        const string& run_id, int64_t version)
            : DatasetIterator<Dataset>(params),
              hash_dir_(hash_dir),
              run_id_(run_id),
              version_(version) {}

        ~SnapshotReaderIterator() override {
          mutex_lock l(mu_);
          cancelled_ = true;
          cond_var_.notify_all();
          while (num_active_threads_ > 0) {
            cond_var_.wait(l);
          }
        }

        Status Initialize(IteratorContext* ctx) override {
          mutex_lock l(mu_);
          thread_pool_ = ctx->CreateThreadPool(kSnapshotReaderWorkerPool,
                                               dataset()->num_reader_threads_);
          run_dir_ = io::JoinPath(hash_dir_, run_id_);
          // Get all the files in the run_dir.
          std::vector<std::string> filenames_str;
          TF_RETURN_IF_ERROR(ctx->env()->GetMatchingPaths(
              absl::StrCat(absl::string_view(run_dir_), "/*"), &filenames_str));
          filenames_.resize(filenames_str.size());
          std::copy(filenames_str.begin(), filenames_str.end(),
                    filenames_.begin());
          if (filenames_.empty()) {
            return errors::NotFound("Could not find any files in dir: ",
                                    run_dir_);
          }

          if (dataset()->shuffle_on_read_) {
            uint64 seed = dataset()->seed_ + dataset()->seed2_;
            if (dataset()->seed_ == 0 && dataset()->seed2_ == 0) {
              seed = random::New64();
            }

            std::mt19937 rng(seed);
            std::shuffle(filenames_.begin(), filenames_.end(), rng);
          } else {
            std::sort(filenames_.begin(), filenames_.end());
          }

          for (auto i = 0; i < dataset()->num_reader_threads_; ++i) {
            curr_filenames_.push_back(GetNextFilename());
          }
          return Status::OK();
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          absl::Time start = absl::Now();
          mutex_lock l(mu_);
          if (!background_threads_started_) {
            for (int i = 0; i < dataset()->num_reader_threads_; ++i) {
              ++num_active_threads_;
              thread_pool_->Schedule(
                  [this, i, env = ctx->env()]() { ReadingFilesLoop(env, i); });
            }
            background_threads_started_ = true;
          }

          // Wait till the buffer has something in it.
          while (!cancelled_ && buffer_.empty() &&
                 !background_threads_finished_) {
            cond_var_.wait(l);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "SnapshotDatasetOp::Dataset::SnapshotReaderIterator::GetNext");
          }

          const auto& stats_aggregator = ctx->stats_aggregator();
          if (stats_aggregator) {
            stats_aggregator->AddScalar(
                absl::StrCat(dataset()->node_name(), kSeparator,
                             kSnapshotReadElements),
                static_cast<float>(num_elements_read_), elements_produced_);
            stats_aggregator->AddScalar(
                absl::StrCat(dataset()->node_name(), kSeparator,
                             "snapshot_reader_buffer_size"),
                static_cast<float>(buffer_.size()), elements_produced_);
          }

          if (!buffer_.empty()) {
            Status s = buffer_.front().status;
            if (s.ok()) {
              *end_of_sequence = false;
              *out_tensors = std::move(buffer_.front().value);

              {
                profiler::TraceMe activity(
                    [&]() {
                      return absl::StrCat(prefix(), kSeparator, kBookkeeping);
                    },
                    profiler::TraceMeLevel::kInfo);
                // Printing some statistics along the way.
                int64_t num_bytes = 0;
                for (int i = 0; i < out_tensors->size(); ++i) {
                  num_bytes += (*out_tensors)[i].TotalBytes();
                }
                absl::Time end = absl::Now();
                absl::Duration d = end - start;
                time_spent_micros_ += absl::ToInt64Microseconds(d);
                kbytes_read_ += static_cast<double>(num_bytes) / 1024.0;
                float read_throughput =
                    (kbytes_read_ / 1024.0) / (time_spent_micros_ / 1000000.0);
                if (stats_aggregator) {
                  stats_aggregator->AddScalar(
                      absl::StrCat(dataset()->node_name(), kSeparator,
                                   kSnapshotReadThroughput),
                      read_throughput, elements_produced_);
                }
                elements_produced_++;
                if (elements_produced_ % 10000 == 0) {
                  LOG(INFO)
                      << "Current read throughput (MBPS): " << read_throughput;
                }
              }
            }
            buffer_.pop_front();
            cond_var_.notify_all();
            return s;
          }

          if (background_threads_finished_) {
            *end_of_sequence = true;
            return Status::OK();
          }

          return errors::Internal("Unreachable point in SnapshotReader");
        }

       protected:
        Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override {
          mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kHashDir), hash_dir_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunId), run_id_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunDir), run_dir_));
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kVersionStr), version_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kFilenames, kSizeSuffix)),
              filenames_.size()));
          for (size_t i = 0; i < filenames_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kFilenames, "[", i, "]")),
                filenames_[i]));
          }
          for (auto i = 0; i < dataset()->num_reader_threads_; ++i) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kCurrentFilenames, "[", i, "]")),
                curr_filenames_[i]));
          }
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kElementsProduced),
                                                 elements_produced_));
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kNextFileIndex), next_file_index_));
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kNumFilesDone), num_files_done_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kNumElementsRead),
                                                 num_elements_read_));
          VLOG(2) << "Saving SnapshotReaderIterator: " << num_elements_read_
                  << "; elements_produced: " << elements_produced_;
          return Status::OK();
        }

        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
          mutex_lock l(mu_);
          tstring hash_dir, run_id, run_dir;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kHashDir), &hash_dir));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kHashDir), &run_id));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kHashDir), &run_dir));
          if (run_dir != run_dir_) {
            LOG(ERROR) << "Restoring read iterator from ckpt with old "
                       << "run_dir: " << run_dir
                       << " but new run_dir is: " << run_dir_
                       << ". We'll now restart snapshot creation.";
            return Status::OK();
          }
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunId), &run_id_));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunDir), &run_dir_));
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kVersionStr), &version_));
          curr_filenames_.clear();
          curr_filenames_.reserve(dataset()->num_reader_threads_);
          for (auto i = 0; i < dataset()->num_reader_threads_; ++i) {
            curr_filenames_.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kCurrentFilenames, "[", i, "]")),
                &curr_filenames_.back()));
          }
          size_t filenames_size;
          {
            int64_t temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kFilenames, kSizeSuffix)), &temp));
            filenames_size = static_cast<size_t>(temp);
          }
          if (filenames_.size() != filenames_size) {
            LOG(ERROR) << "Old filenames size: " << filenames_size
                       << "; new filenames size: " << filenames_.size();
          }
          filenames_.clear();
          filenames_.reserve(filenames_size);
          for (size_t i = 0; i < filenames_size; ++i) {
            filenames_.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kFilenames, "[", i, "]")),
                &filenames_.back()));
          }
          {
            int64_t temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kElementsProduced), &temp));
            elements_produced_ = static_cast<uint64>(temp);
          }
          {
            int64_t temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kNextFileIndex), &temp));
            next_file_index_ = static_cast<uint64>(temp);
          }
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kNumFilesDone), &num_files_done_));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumElementsRead),
                                                &num_elements_read_));
          VLOG(2) << "Restoring SnapshotReaderIterator: " << num_elements_read_
                  << "; elements_produced: " << elements_produced_;
          return Status::OK();
        }

       private:
        // Reads one file end to end.
        Status ReadFile(Env* env, const string& filename) {
          std::unique_ptr<snapshot_util::Reader> reader;
          TF_RETURN_IF_ERROR(snapshot_util::Reader::Create(
              env, filename, dataset()->compression_, version_,
              dataset()->output_dtypes(), &reader));
          while (true) {
            // Wait for a slot in the buffer.
            {
              mutex_lock l(mu_);
              while (!cancelled_ &&
                     buffer_.size() >= dataset()->reader_buffer_size_) {
                cond_var_.wait(l);
              }

              if (cancelled_) {
                return errors::Cancelled(
                    "SnapshotDatasetOp::Dataset::SnapshotReaderIterator::"
                    "ReadFile");
              }
            }
            std::vector<Tensor> read_tensors;
            Status s = reader->ReadTensors(&read_tensors);
            if (s.ok()) {
              profiler::TraceMe activity(
                  [&]() { return absl::StrCat(prefix(), kSeparator, kParse); },
                  profiler::TraceMeLevel::kInfo);
              BufferElement elem;
              elem.value = std::move(read_tensors);
              elem.status = Status::OK();
              mutex_lock l(mu_);
              buffer_.push_back(std::move(elem));
              num_elements_read_++;
              cond_var_.notify_all();
            } else if (errors::IsOutOfRange(s)) {
              return Status::OK();
            } else {
              return s;
            }
          }
          return Status::OK();
        }

        string GetNextFilename() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          if (next_file_index_ >= filenames_.size()) {
            return "";
          }
          string filename = io::JoinPath(dataset()->reader_path_prefix_,
                                         filenames_[next_file_index_]);
          next_file_index_++;
          return filename;
        }

        // Pulls one file off the filenames_ list and reads it through. When
        // all files are read, terminates.
        void ReadingFilesLoop(Env* env, int i) {
          auto cleanup = gtl::MakeCleanup([this]() {
            mutex_lock l(mu_);
            --num_active_threads_;
            cond_var_.notify_all();
          });
          while (true) {
            string filename = "";
            {
              mutex_lock l(mu_);
              filename = curr_filenames_[i];
              if (filename.empty()) {
                return;
              }
              VLOG(2) << "Starting to read: " << filename;
            }
            Status s = ReadFile(env, filename);
            // If we get to the end of the file, it's a clean termination and
            // we are at the end of the file. If all files have been processed,
            // then we insert an end_of_sequence marker in the buffer and
            // terminate the loop.
            if (s.ok()) {
              VLOG(2) << "Finished reading: " << filename;
              mutex_lock l(mu_);
              num_files_done_++;
              if (num_files_done_ >= filenames_.size()) {
                background_threads_finished_ = true;
                cond_var_.notify_all();
                return;
              }
              curr_filenames_[i] = GetNextFilename();
            } else {
              LOG(ERROR) << "Encountered an error: " << s.ToString();
              BufferElement elem;
              elem.status = s;
              mutex_lock l(mu_);
              buffer_.push_back(std::move(elem));
              cond_var_.notify_all();
              return;
            }
          }
        }

        Status WriteStatus(IteratorStateWriter* writer, size_t index,
                           const Status& status)
            TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              CodeKey(index), static_cast<int64_t>(status.code())));
          if (!status.ok()) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                                   status.error_message()));
          }
          return Status::OK();
        }

        Status ReadStatus(IteratorStateReader* reader, size_t index,
                          Status* status) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
          int64_t code_int;
          TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
          error::Code code = static_cast<error::Code>(code_int);

          if (code != error::Code::OK) {
            tstring error_message;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(ErrorMessageKey(index), &error_message));
            *status = Status(code, error_message);
          } else {
            *status = Status::OK();
          }
          return Status::OK();
        }

        string CodeKey(size_t index) {
          return full_name(strings::StrCat(kStatus, "[", index, "]", kCode));
        }

        string ErrorMessageKey(size_t index) {
          return full_name(
              strings::StrCat(kStatus, "[", index, "]", kErrorMessage));
        }

        struct BufferElement {
          Status status;
          std::vector<Tensor> value;
        };

        mutex mu_;
        condition_variable cond_var_;

        const string hash_dir_;
        tstring run_id_ TF_GUARDED_BY(mu_);
        tstring run_dir_ TF_GUARDED_BY(mu_);
        int64_t version_;
        std::vector<tstring> filenames_;

        uint64 elements_produced_ TF_GUARDED_BY(mu_) = 0;
        int64_t time_spent_micros_ TF_GUARDED_BY(mu_) = 0;
        double kbytes_read_ TF_GUARDED_BY(mu_) = 0;
        size_t next_file_index_ TF_GUARDED_BY(mu_) = 0;
        int64_t num_files_done_ TF_GUARDED_BY(mu_) = 0;

        std::unique_ptr<thread::ThreadPool> thread_pool_;
        int64_t num_active_threads_ TF_GUARDED_BY(mu_) = 0;
        std::deque<BufferElement> buffer_ TF_GUARDED_BY(mu_);
        bool cancelled_ TF_GUARDED_BY(mu_) = false;
        bool background_threads_started_ TF_GUARDED_BY(mu_) = false;
        bool background_threads_finished_ TF_GUARDED_BY(mu_) = false;
        int64_t num_elements_read_ TF_GUARDED_BY(mu_) = 0;
        // curr_filenames_ tracks which file is being read by each thread.
        std::vector<tstring> curr_filenames_ TF_GUARDED_BY(mu_);
      };

      class SnapshotWriterIterator : public DatasetIterator<Dataset> {
       public:
        static constexpr const char* const kProcessOneElement =
            "ProcessOneElement";

        explicit SnapshotWriterIterator(const Params& params,
                                        const string& hash_dir,
                                        const string& run_id)
            : DatasetIterator<Dataset>(params),
              hash_dir_(hash_dir),
              run_id_(run_id) {
          if (run_id_.empty()) {
            run_id_ = strings::StrCat(
                strings::Hex(random::New64(), strings::kZeroPad4));
          }
          run_dir_ =
              io::JoinPath(dataset()->writer_path_prefix_, hash_dir_, run_id_);
        }

        ~SnapshotWriterIterator() override {
          mutex_lock l(mu_);
          cancelled_ = true;
          cond_var_.notify_all();
          while (num_active_threads_ > 0) {
            cond_var_.wait(l);
          }
        }

        Status Initialize(IteratorContext* ctx) override {
          thread_pool_ = ctx->CreateThreadPool(kSnapshotWriterWorkerPool,
                                               dataset()->num_writer_threads_);
          return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                                 &input_impl_);
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          absl::Time start = absl::Now();

          bool first_call;
          bool is_restored;
          {
            mutex_lock l(mu_);
            first_call = first_call_;
            is_restored = is_restored_;
            if (first_call_) {
              // If we're restoring then the directory already exists and we
              // don't want to overwrite the snapshot metadata file.
              if (!is_restored_) {
                TF_RETURN_IF_ERROR(ctx->env()->RecursivelyCreateDir(run_dir_));
                experimental::SnapshotMetadataRecord metadata;
                metadata.set_creation_timestamp(EnvTime::NowMicros());
                metadata.set_graph_hash(dataset()->graph_hash_);
                metadata.set_run_id(run_id_.data(), run_id_.size());
                metadata.set_version(kCurrentVersion);
                for (const auto& output_dtype : dataset()->output_dtypes()) {
                  metadata.add_dtype(output_dtype);
                }
                metadata.set_finalized(false);
                TF_RETURN_IF_ERROR(snapshot_util::WriteMetadataFile(
                    ctx->env(), hash_dir_, &metadata));
              }
              for (int i = 0; i < dataset()->num_writer_threads_; ++i) {
                ++num_active_threads_;
                thread_pool_->Schedule(
                    [this, env = ctx->env()]() { WriterThread(env); });
              }
              first_call_ = false;
            }
          }

          // When we reach the end of the data, we'd like to finalize the
          // snapshot and write the metadata file out. If we just check for
          // end_of_sequence on the GetNext call then we will need to make
          // N + 1 GetNext calls (if N is the total number of elements in the
          // dataset). So right now we solve this issue by prefetching the next
          // element in the data stream. Therefore the first call ends up
          // pulling two elements.
          if (first_call && !is_restored) {
            TF_RETURN_IF_ERROR(FillBuffer(ctx));
          }

          {
            mutex_lock l(mu_);
            // Populate out_tensors with the prefetched data.
            *out_tensors = std::move(next_elem_.value);
            *end_of_sequence = next_elem_.end_of_sequence;
          }

          // Update prefetched_elem with the next element.
          TF_RETURN_IF_ERROR(FillBuffer(ctx));

          {
            profiler::TraceMe activity(
                [&]() {
                  return absl::StrCat(prefix(), kSeparator, kBookkeeping);
                },
                profiler::TraceMeLevel::kInfo);

            // Book keeping to report some statistics.
            mutex_lock l(mu_);
            int64_t num_bytes = 0;
            for (const auto& out_tensor : *out_tensors) {
              num_bytes += out_tensor.TotalBytes();
            }

            const auto& stats_aggregator = ctx->stats_aggregator();
            if (stats_aggregator) {
              stats_aggregator->AddScalar(
                  absl::StrCat(dataset()->node_name(), kSeparator,
                               kSnapshotWrittenElements),
                  static_cast<float>(num_elements_written_),
                  elements_produced_);
              stats_aggregator->AddScalar(
                  absl::StrCat(dataset()->node_name(), kSeparator,
                               "snapshot_writer_buffer_size"),
                  static_cast<float>(buffer_.size()), elements_produced_);
            }

            absl::Time end = absl::Now();
            absl::Duration d = end - start;
            time_spent_micros_ += absl::ToInt64Microseconds(d);
            bytes_produced_ += num_bytes;
            float write_throughput = (bytes_produced_ * 1000000.0) /
                                     (time_spent_micros_ * 1024.0 * 1024.0);
            if (stats_aggregator) {
              stats_aggregator->AddScalar(
                  absl::StrCat(dataset()->node_name(), kSeparator,
                               kSnapshotWriteThroughput),
                  write_throughput, elements_produced_);
            }

            elements_produced_++;
            if (elements_produced_ % 10000 == 0) {
              LOG(INFO) << "Current write throughput (MBPS): "
                        << write_throughput;
            }
          }
          return Status::OK();
        }

       protected:
        Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override {
          mutex_lock l(mu_);
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
          if (end_of_sequence_) {
            TF_RETURN_IF_ERROR(
                writer->WriteScalar(full_name(kEndOfSequence), ""));
          }
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name(kHashDir), hash_dir_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunId), run_id_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kRunDir), run_dir_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kElementsProduced),
                                                 elements_produced_));
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kBuffer, kSizeSuffix)),
              buffer_.size()));
          for (size_t i = 0; i < buffer_.size(); ++i) {
            auto& buffer_element = buffer_[i];
            if (buffer_element.end_of_sequence) {
              TF_RETURN_IF_ERROR(writer->WriteScalar(
                  full_name(
                      strings::StrCat(kBuffer, "[", i, "].", kEndOfSequence)),
                  ""));
            }
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
                buffer_element.value.size()));
            for (size_t j = 0; j < buffer_element.value.size(); j++) {
              TF_RETURN_IF_ERROR(writer->WriteTensor(
                  full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
                  buffer_element.value[j]));
            }
          }
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kNumElementsWritten),
                                                 num_elements_written_));
          if (next_elem_.end_of_sequence) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat(kNextElem, ".", kEndOfSequence)),
                ""));
          }
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat(kNextElem, kSizeSuffix)),
              next_elem_.value.size()));
          for (size_t i = 0; i < next_elem_.value.size(); i++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat(kNextElem, "[", i, "]")),
                next_elem_.value[i]));
          }
          VLOG(2) << "Saving SnapshotWriterIterator: " << num_elements_written_
                  << "; elements_produced: " << elements_produced_;
          return Status::OK();
        }

        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
          mutex_lock l(mu_);
          buffer_.clear();
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
          tstring hash_dir;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name(kHashDir), &hash_dir));
          // If the hash dir has changed then we restart writing.
          if (hash_dir != hash_dir_) {
            LOG(INFO) << "Old hash dir from ckpt: " << hash_dir
                      << " is not the same as the new one: " << hash_dir_;
            return Status::OK();
          }
          is_restored_ = true;
          if (reader->Contains(full_name(kEndOfSequence))) {
            end_of_sequence_ = true;
          } else {
            end_of_sequence_ = false;
          }
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunId), &run_id_));
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kRunDir), &run_dir_));
          {
            int64_t temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name(kElementsProduced), &temp));
            elements_produced_ = static_cast<uint64>(temp);
          }
          size_t buffer_size;
          {
            int64_t temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kBuffer, kSizeSuffix)), &temp));
            buffer_size = static_cast<size_t>(temp);
          }
          for (size_t i = 0; i < buffer_size; i++) {
            buffer_.emplace_back();
            auto& buffer_element = buffer_.back();
            size_t value_size;
            {
              int64_t temp;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
                  &temp));
              value_size = static_cast<size_t>(temp);
            }
            if (reader->Contains(full_name(
                    strings::StrCat(kBuffer, "[", i, "].", kEndOfSequence)))) {
              buffer_element.end_of_sequence = true;
            } else {
              buffer_element.end_of_sequence = false;
            }
            buffer_element.value.reserve(value_size);
            for (size_t j = 0; j < value_size; j++) {
              buffer_element.value.emplace_back();
              TF_RETURN_IF_ERROR(reader->ReadTensor(
                  ctx->flr(),
                  full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
                  &buffer_element.value.back()));
            }
          }
          // Since the last save we might have written out some files. So we
          // get a list of files in the directory and take the final filename
          // written. We use the name of the snapshot file to figure out
          // next_file_index_;
          std::vector<std::string> filenames;
          TF_RETURN_IF_ERROR(ctx->env()->GetMatchingPaths(
              absl::StrCat(absl::string_view(run_dir_), "/*"), &filenames));
          std::sort(filenames.begin(), filenames.end());
          std::string final_filename = filenames.back();
          std::vector<std::string> split_filename =
              absl::StrSplit(final_filename, '/');
          std::vector<std::string> split_snapshot_filename =
              absl::StrSplit(split_filename.back(), '.');
          std::string max_num_str = split_snapshot_filename[0];
          uint64 max_num;
          if (!strings::safe_strtou64(max_num_str, &max_num)) {
            return errors::Internal("Could not parse: ", max_num, " as uint64");
          }
          next_file_index_ = max_num + 1;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumElementsWritten),
                                                &num_elements_written_));
          size_t next_elem_size;
          {
            int64_t temp;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat(kNextElem, kSizeSuffix)), &temp));
            next_elem_size = static_cast<size_t>(temp);
          }
          if (reader->Contains(
                  full_name(strings::StrCat(kNextElem, ".", kEndOfSequence)))) {
            next_elem_.end_of_sequence = true;
          } else {
            next_elem_.end_of_sequence = false;
          }
          next_elem_.value.reserve(next_elem_size);
          for (size_t i = 0; i < next_elem_size; i++) {
            next_elem_.value.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), full_name(strings::StrCat(kNextElem, "[", i, "]")),
                &next_elem_.value.back()));
          }
          VLOG(2) << "Restoring SnapshotWriterIterator: "
                  << num_elements_written_
                  << "; elements_produced: " << elements_produced_;
          return Status::OK();
        }

       private:
        string GetSnapshotFilename() {
          mutex_lock l(mu_);
          string snapshot_data_filename = io::JoinPath(
              run_dir_, strings::Printf(
                            "%08llu.snapshot",
                            static_cast<unsigned long long>(next_file_index_)));
          next_file_index_++;
          return snapshot_data_filename;
        }

        Status FillBuffer(IteratorContext* ctx) TF_LOCKS_EXCLUDED(mu_) {
          snapshot_util::ElementOrEOF elem;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &elem.value, &elem.end_of_sequence));

          mutex_lock l(mu_);
          next_elem_ = std::move(elem);

          if (next_elem_.end_of_sequence) {
            end_of_sequence_ = true;
            cond_var_.notify_all();
            // Now we wait till all background threads finish.
            while (num_active_threads_ > 0) {
              cond_var_.wait(l);
            }
            return Status::OK();
          }

          // Wait for a space in the buffer_.
          while (!cancelled_ &&
                 buffer_.size() >= dataset()->writer_buffer_size_) {
            cond_var_.wait(l);
          }

          if (cancelled_) {
            return errors::Cancelled(
                "SnapshotDatasetOp::SnapshotWriterIterator::GetNext");
          }

          if (buffer_.size() >= dataset()->writer_buffer_size_) {
            return errors::Internal(
                "Buffer size: ", buffer_.size(), " should be smaller than ",
                "maximum size: ", dataset()->writer_buffer_size_);
          }

          snapshot_util::ElementOrEOF elem_copy = next_elem_;
          buffer_.push_back(elem_copy);
          cond_var_.notify_all();
          return Status::OK();
        }

        Status ProcessOneElement(Env* env, int64_t* bytes_written,
                                 string* snapshot_data_filename,
                                 std::unique_ptr<snapshot_util::Writer>* writer,
                                 bool* end_of_processing) {
          profiler::TraceMe activity(
              [&]() {
                return absl::StrCat(prefix(), kSeparator, kProcessOneElement);
              },
              profiler::TraceMeLevel::kInfo);
          bool cancelled = false;
          *end_of_processing = false;
          bool produced_elem = false;
          bool snapshot_failed = false;
          snapshot_util::ElementOrEOF elem;
          {
            mutex_lock l(mu_);
            // Wait for buffer to not be empty.
            while (!cancelled_ && buffer_.empty() && !end_of_sequence_ &&
                   !snapshot_failed_) {
              cond_var_.wait(l);
            }
            cancelled = cancelled_;
            if (!buffer_.empty()) {
              produced_elem = true;
              std::swap(elem, buffer_.front());
              buffer_.pop_front();
              cond_var_.notify_all();
            } else {
              *end_of_processing = end_of_sequence_;
            }
            snapshot_failed = snapshot_failed_;
          }

          if (cancelled || snapshot_failed) {
            TF_RETURN_IF_ERROR((*writer)->Close());
            if (snapshot_failed) {
              return errors::Internal(
                  "SnapshotDataset::SnapshotWriterIterator snapshot failed");
            }
            return errors::Cancelled(
                "SnapshotDataset::SnapshotWriterIterator cancelled");
          }

          if (produced_elem) {
            for (const auto& out_tensor : elem.value) {
              *bytes_written += out_tensor.TotalBytes();
            }

            bool should_close;
            TF_RETURN_IF_ERROR(
                ShouldCloseWriter(env, *snapshot_data_filename, *bytes_written,
                                  (*writer).get(), &should_close));
            if (should_close) {
              // If we exceed the shard size, we get a new file and reset.
              TF_RETURN_IF_ERROR((*writer)->Close());
              *snapshot_data_filename = GetSnapshotFilename();

              TF_RETURN_IF_ERROR(snapshot_util::Writer::Create(
                  env, *snapshot_data_filename, dataset()->compression_,
                  kCurrentVersion, dataset()->output_dtypes(), writer));
              *bytes_written = 0;
            }
            TF_RETURN_IF_ERROR((*writer)->WriteTensors(elem.value));
            return Status::OK();
          }

          if (*end_of_processing) {
            TF_RETURN_IF_ERROR((*writer)->Close());
            mutex_lock l(mu_);
            if (!written_final_metadata_file_) {
              experimental::SnapshotMetadataRecord metadata;
              bool file_exists;
              TF_RETURN_IF_ERROR(snapshot_util::ReadMetadataFile(
                  env, hash_dir_, &metadata, &file_exists));

              if (metadata.run_id() == run_id_) {
                metadata.set_finalized(true);
                TF_RETURN_IF_ERROR(snapshot_util::WriteMetadataFile(
                    env, hash_dir_, &metadata));
              } else {
                // TODO(frankchn): We lost the race, remove all snapshots.
              }
              written_final_metadata_file_ = true;
              cond_var_.notify_all();
            }
          }
          return Status::OK();
        }

        // Just pulls off elements from the buffer and writes them.
        void WriterThread(Env* env) {
          auto cleanup = gtl::MakeCleanup([this]() {
            mutex_lock l(mu_);
            --num_active_threads_;
            cond_var_.notify_all();
          });

          int64_t bytes_written = 0;
          string snapshot_data_filename = GetSnapshotFilename();
          std::unique_ptr<snapshot_util::Writer> writer;
          Status s = snapshot_util::Writer::Create(
              env, snapshot_data_filename, dataset()->compression_,
              kCurrentVersion, dataset()->output_dtypes(), &writer);
          if (!s.ok()) {
            LOG(ERROR) << "Creating " << snapshot_data_filename
                       << " failed: " << s.ToString();
            mutex_lock l(mu_);
            snapshot_failed_ = true;
            cond_var_.notify_all();
            return;
          }

          bool end_of_processing = false;
          while (!end_of_processing) {
            Status s =
                ProcessOneElement(env, &bytes_written, &snapshot_data_filename,
                                  &writer, &end_of_processing);
            if (!s.ok()) {
              LOG(INFO) << "Error while writing snapshot data to disk: "
                        << s.ToString();
              mutex_lock l(mu_);
              snapshot_failed_ = true;
              cond_var_.notify_all();
              return;
            }
            mutex_lock l(mu_);
            num_elements_written_++;
          }
        }

        Status ShouldCloseWriter(Env* env, const string& filename,
                                 uint64 bytes_written,
                                 snapshot_util::Writer* writer,
                                 bool* should_close) {
          // If the compression ratio has been estimated, use it to decide
          // whether the file should be closed. We avoid estimating the
          // compression ratio repeatedly because it requires syncing the file,
          // which can be expensive.
          {
            tf_shared_lock l(mu_);
            if (compression_ratio_ > 0.0) {
              *should_close = bytes_written > (compression_ratio_ *
                                               dataset()->shard_size_bytes_);
              return Status::OK();
            }
          }
          // If the number of bytes written aren't shard_size_bytes_ yet, we
          // keep on going.
          if (bytes_written <= dataset()->shard_size_bytes_) {
            *should_close = false;
            return Status::OK();
          }
          // Use the actual file size to determine compression ratio.
          // Make sure that all bytes are written out.
          TF_RETURN_IF_ERROR(writer->Sync());
          uint64 file_size;
          TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
          mutex_lock l(mu_);
          compression_ratio_ = static_cast<double>(bytes_written) /
                               static_cast<double>(file_size);
          LOG(INFO) << "Writing compression achieved: " << compression_ratio_;
          *should_close = file_size >= dataset()->shard_size_bytes_;
          return Status::OK();
        }

        mutex mu_;
        // This condition variable is notified
        // 1. By the background writer threads when an element from the buffer
        //    is consumed.
        // 2. By the main thread when it puts something into the buffer.
        // 3. By the main thread when the destructor is called to cancel.
        // 4. By the background writer threads when any error is encountered
        //    while writing.
        // 5. By the background threads when they finish.
        condition_variable cond_var_;

        snapshot_util::ElementOrEOF next_elem_ TF_GUARDED_BY(mu_);
        std::unique_ptr<IteratorBase> input_impl_;

        const string hash_dir_;
        tstring run_id_ TF_GUARDED_BY(mu_);
        tstring run_dir_ TF_GUARDED_BY(mu_);
        double compression_ratio_ TF_GUARDED_BY(mu_) = 0.0;
        bool is_restored_ TF_GUARDED_BY(mu_) = false;

        uint64 elements_produced_ TF_GUARDED_BY(mu_) = 0;
        int64_t time_spent_micros_ TF_GUARDED_BY(mu_) = 0;
        int64_t bytes_produced_ TF_GUARDED_BY(mu_) = 0;

        std::deque<snapshot_util::ElementOrEOF> buffer_ TF_GUARDED_BY(mu_);
        bool snapshot_failed_ TF_GUARDED_BY(mu_) = false;
        bool cancelled_ TF_GUARDED_BY(mu_) = false;
        bool first_call_ TF_GUARDED_BY(mu_) = true;
        bool end_of_sequence_ TF_GUARDED_BY(mu_) = false;
        bool written_final_metadata_file_ TF_GUARDED_BY(mu_) = false;
        uint64 next_file_index_ TF_GUARDED_BY(mu_) = 0;
        std::unique_ptr<thread::ThreadPool> thread_pool_;
        int64_t num_active_threads_ TF_GUARDED_BY(mu_) = 0;
        int64_t num_elements_written_ = 0;
      };

      class SnapshotPassthroughIterator : public DatasetIterator<Dataset> {
       public:
        explicit SnapshotPassthroughIterator(const Params& params)
            : DatasetIterator<Dataset>(params) {}

        Status Initialize(IteratorContext* ctx) override {
          return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                                 &input_impl_);
        }

        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override {
          return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
        }

       protected:
        Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override {
          return SaveInput(ctx, writer, input_impl_);
        }

        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
          return RestoreInput(ctx, reader, input_impl_);
        }

       private:
        std::unique_ptr<IteratorBase> input_impl_;
      };

      string hash_dir_ TF_GUARDED_BY(mu_);
      snapshot_util::Mode state_ TF_GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> iterator_ TF_GUARDED_BY(mu_);

      mutex mu_;
    };

    const DatasetBase* const input_;
    const tstring dir_;
    const string graph_hash_;

    const string reader_path_prefix_;
    const string writer_path_prefix_;
    const string compression_;

    const uint64 shard_size_bytes_;
    const uint64 pending_snapshot_expiry_seconds_;
    const uint64 num_reader_threads_;
    const uint64 reader_buffer_size_;
    const uint64 num_writer_threads_;
    const uint64 writer_buffer_size_;
    const bool shuffle_on_read_;

    const uint64 seed_;
    const uint64 seed2_;

    const std::string mode_;
    const std::string snapshot_name_;
  };

  Status ComputeDatasetHash(const GraphDef& graph_def, const std::string& path,
                            uint64* hash) {
    TF_RETURN_IF_ERROR(HashGraph(graph_def, hash));
    // Adding path, compression, reader / writer path prefix, shard size
    // bytes to the fp as they effect the data written on disk.
    *hash = Hash64Combine(*hash, Hash64(path));
    *hash = Hash64Combine(*hash, Hash64(compression_));
    *hash = Hash64Combine(*hash, Hash64(reader_path_prefix_));
    *hash = Hash64Combine(*hash, Hash64(writer_path_prefix_));
    *hash = Hash64Combine(*hash, shard_size_bytes_);
    return Status::OK();
  }

  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;

  string reader_path_prefix_;
  string writer_path_prefix_;
  string compression_;

  int64_t shard_size_bytes_;
  int64_t pending_snapshot_expiry_seconds_;
  int64_t num_reader_threads_;
  int64_t reader_buffer_size_;
  int64_t num_writer_threads_;
  int64_t writer_buffer_size_;
  bool shuffle_on_read_;

  int64_t seed_;
  int64_t seed2_;

  std::string mode_;
  std::string snapshot_name_;
};

REGISTER_KERNEL_BUILDER(Name("SnapshotDataset").Device(DEVICE_CPU),
                        SnapshotDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
