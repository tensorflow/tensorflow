/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <sys/stat.h>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/experimental/lmdb_dataset_op.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/platform/file_system.h"

#include "lmdb.h"  // NOLINT(build/include)

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const LMDBDatasetOp::kDatasetType;
/* static */ constexpr const char* const LMDBDatasetOp::kFileNames;
/* static */ constexpr const char* const LMDBDatasetOp::kOutputTypes;
/* static */ constexpr const char* const LMDBDatasetOp::kOutputShapes;



class LMDBDatasetOp::Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const std::vector<string>& filenames)
        : DatasetBase(DatasetContext(ctx)), filenames_(filenames) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::LMDB")});
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes =
          new DataTypeVector({DT_STRING, DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}, {}});
      return *shapes;
    }

    string DebugString() const override { return "LMDBDatasetOp::Dataset"; }

    Status CheckExternalState() const override { return Status::OK(); }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* filenames = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {filenames}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      virtual ~Iterator() {
        // Close any open database connections.
        ResetStreamsLocked();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        do {
          if (mdb_cursor_) {
            out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                      TensorShape({}));
            Tensor& key_tensor = out_tensors->back();
            key_tensor.scalar<tstring>()() = string(
                static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);

            out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                      TensorShape({}));
            Tensor& value_tensor = out_tensors->back();
            value_tensor.scalar<tstring>()() =
                string(static_cast<const char*>(mdb_value_.mv_data),
                       mdb_value_.mv_size);

            int val;
            val = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT);
            if (val != MDB_SUCCESS && val != MDB_NOTFOUND) {
              return errors::InvalidArgument(mdb_strerror(val));
            }
            if (val == MDB_NOTFOUND) {
              ResetStreamsLocked();
              ++current_file_index_;
            }
            *end_of_sequence = false;
            return Status::OK();
          }
          if (current_file_index_ == dataset()->filenames_.size()) {
            *end_of_sequence = true;
            ResetStreamsLocked();
            return Status::OK();
          }

          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        } while (true);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(IteratorStateWriter* writer) override {
        return errors::Unimplemented(
            "Checkpointing is currently not supported for LMDBDataset.");
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        return errors::Unimplemented(
            "Checkpointing is currently not supported for LMDBDataset.");
      }

     private:
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (current_file_index_ >= dataset()->filenames_.size()) {
          return errors::InvalidArgument(
              "current_file_index_:", current_file_index_,
              " >= filenames_.size():", dataset()->filenames_.size());
        }
        const string& filename = dataset()->filenames_[current_file_index_];

        int val = mdb_env_create(&mdb_env_);
        if (val != MDB_SUCCESS) {
          return errors::InvalidArgument(mdb_strerror(val));
        }
        int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;

        struct stat source_stat;
        if (stat(filename.c_str(), &source_stat) == 0 &&
            (source_stat.st_mode & S_IFREG)) {
          flags |= MDB_NOSUBDIR;
        }
        val = mdb_env_open(mdb_env_, filename.c_str(), flags, 0664);
        if (val != MDB_SUCCESS) {
          return errors::InvalidArgument(mdb_strerror(val));
        }
        val = mdb_txn_begin(mdb_env_, nullptr, MDB_RDONLY, &mdb_txn_);
        if (val != MDB_SUCCESS) {
          return errors::InvalidArgument(mdb_strerror(val));
        }
        val = mdb_dbi_open(mdb_txn_, nullptr, 0, &mdb_dbi_);
        if (val != MDB_SUCCESS) {
          return errors::InvalidArgument(mdb_strerror(val));
        }
        val = mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
        if (val != MDB_SUCCESS) {
          return errors::InvalidArgument(mdb_strerror(val));
        }
        val = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
        if (val != MDB_SUCCESS && val != MDB_NOTFOUND) {
          return errors::InvalidArgument(mdb_strerror(val));
        }
        if (val == MDB_NOTFOUND) {
          ResetStreamsLocked();
        }
        return Status::OK();
      }
      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (mdb_env_ != nullptr) {
          if (mdb_cursor_) {
            mdb_cursor_close(mdb_cursor_);
            mdb_cursor_ = nullptr;
          }
          mdb_dbi_close(mdb_env_, mdb_dbi_);
          mdb_txn_abort(mdb_txn_);
          mdb_env_close(mdb_env_);
          mdb_txn_ = nullptr;
          mdb_dbi_ = 0;
          mdb_env_ = nullptr;
        }
      }
      mutex mu_;
      size_t current_file_index_ GUARDED_BY(mu_) = 0;
      MDB_env* mdb_env_ GUARDED_BY(mu_) = nullptr;
      MDB_txn* mdb_txn_ GUARDED_BY(mu_) = nullptr;
      MDB_dbi mdb_dbi_ GUARDED_BY(mu_) = 0;
      MDB_cursor* mdb_cursor_ GUARDED_BY(mu_) = nullptr;

      MDB_val mdb_key_ GUARDED_BY(mu_);
      MDB_val mdb_value_ GUARDED_BY(mu_);
    };

    const std::vector<string> filenames_;
  };

void LMDBDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input("filenames", &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  std::vector<string> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
  }

  *output = new Dataset(ctx, filenames);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("LMDBDataset").Device(DEVICE_CPU), LMDBDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalLMDBDataset").Device(DEVICE_CPU),
                        LMDBDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
