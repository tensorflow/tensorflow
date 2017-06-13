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

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/dataset.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level description of
// the following op.

class CacheDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit CacheDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    // Parse out the filenames tensor.
    string filename;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "filename", &filename));

    if (filename.empty()) {
      *output = new MemoryDataset(input);
    } else {
      *output = new FileDataset(input, filename, ctx->env());
    }
  }

 private:
  class FileDataset : public DatasetBase {
   public:
    explicit FileDataset(const DatasetBase* input, string filename, Env* env)
        : input_(input),
          filename_(std::move(filename)),
          env_(env),
          num_tensors_(input->output_dtypes().size()),
          tensor_index_padding_size_(StringPaddingSize(num_tensors_)),
          item_index_padding_size_(StringPaddingSize(kMaxItems)),
          tensor_format_string_(strings::Printf("%%%zuzu_%%%zuzu",
                                                item_index_padding_size_,
                                                tensor_index_padding_size_)) {
      input_->Ref();
      DCHECK_EQ(item_index_padding_size_, 7);
    }

    ~FileDataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      if (env_->FileExists(strings::StrCat(filename_, ".index")).ok()) {
        return std::unique_ptr<IteratorBase>(new FileReaderIterator(this));
      } else {
        return std::unique_ptr<IteratorBase>(new FileWriterIterator(this));
      }
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() override { return "CacheDatasetOp::FileDataset"; }

   private:
    static size_t StringPaddingSize(size_t num_tensors) {
      return strings::Printf("%zu", num_tensors - 1).size();
    }

    string FormatName(size_t item_index, size_t tensor_index) const {
      return strings::Printf(tensor_format_string_.c_str(), item_index,
                             tensor_index);
    }

    // FileWriterIterator passes through and caches items from the input
    // FileDataset.
    //
    // This iterator is used when the cache directory is not found on disk. It
    // creates the cache directory, and passes on the underlying iterator's
    // elements.
    class FileWriterIterator : public DatasetIterator<FileDataset> {
     public:
      explicit FileWriterIterator(const FileDataset* dataset)
          : DatasetIterator<FileDataset>(dataset),
            cur_index_(0),
            input_impl_(dataset->input_->MakeIterator()),
            writer_(dataset->env_, dataset->filename_),
            lockfile_(strings::StrCat(dataset->filename_, ".lockfile")),
            lockfile_created_(false),
            iteration_completed_(false) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsureLockFileExists());
        TF_RETURN_IF_ERROR(writer_.status());
        if (cur_index_ >= kMaxItems) {
          // As a courtesy, close the [truncated] cache file.
          Status s = Finish();
          if (!s.ok()) {
            LOG(ERROR) << s;
          }
          return errors::InvalidArgument(
              "Upstream iterator is producing more than ", kMaxItems,
              " items, which is more than the cache limit.");
        }

        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence && out_tensors->empty()) {
          TF_RETURN_IF_ERROR(Finish());
          cur_index_++;
          return Status::OK();
        }
        if (out_tensors->size() != dataset()->num_tensors_) {
          return errors::Internal(
              "Upstream iterator returned invalid number of tensors. Expected ",
              dataset()->num_tensors_, " got: ", out_tensors->size());
        }
        size_t tensor_index = 0;
        for (const Tensor& t : *out_tensors) {
          DCHECK_LT(tensor_index, dataset()->num_tensors_);
          string key = dataset()->FormatName(cur_index_, tensor_index++);
          TF_RETURN_IF_ERROR(writer_.Add(key, t));
        }
        if (*end_of_sequence) {
          TF_RETURN_IF_ERROR(Finish());
        }
        cur_index_++;
        return Status::OK();
      }

     private:
      Status EnsureLockFileExists() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        if (iteration_completed_)
          return errors::OutOfRange(
              "Attempting to call get_next after iteration should have "
              "finished.");
        if (lockfile_created_ && !iteration_completed_) return Status::OK();
        // Perform rudimentary locking to help catch concurrent writes to the
        // same cache files.
        if (dataset()->env_->FileExists(lockfile_).ok()) {
          // Attempt to read the contents of the lockfile.
          char contents_scratch[151] = {0};  // Initialize all to 0.
          StringPiece contents;
          std::unique_ptr<RandomAccessFile> file;
          if (dataset()->env_->NewRandomAccessFile(lockfile_, &file).ok()) {
            file->Read(0, 150, &contents, contents_scratch).IgnoreError();
          }
          return errors::AlreadyExists(
              "There appears to be a concurrent caching iterator running - "
              "cache lockfile already exists ('",
              lockfile_,
              "'). If you are sure no other running TF computations are using "
              "this cache prefix, delete the lockfile and re-initialize the "
              "iterator. Lockfile contents: ",
              contents);
        } else {
          // Create the file, and write some basic contents.
          std::unique_ptr<WritableFile> lockfile;
          TF_RETURN_IF_ERROR(
              dataset()->env_->NewWritableFile(lockfile_, &lockfile));
          TF_RETURN_IF_ERROR(lockfile->Append(
              strings::StrCat("Created at: ", dataset()->env_->NowSeconds())));
          lockfile_created_ = true;
          return Status::OK();
        }
      }

      Status Finish() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        iteration_completed_ = true;
        TF_RETURN_IF_ERROR(writer_.Finish());
        TF_RETURN_IF_ERROR(dataset()->env_->DeleteFile(lockfile_));
        return Status::OK();
      }

      mutex mu_;
      size_t cur_index_ GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      BundleWriter writer_ GUARDED_BY(mu_);
      const string lockfile_;
      bool lockfile_created_ GUARDED_BY(mu_);
      bool iteration_completed_ GUARDED_BY(mu_);
    };  // FileWriterIterator

    class FileReaderIterator : public DatasetIterator<FileDataset> {
     public:
      explicit FileReaderIterator(const FileDataset* dataset)
          : DatasetIterator<FileDataset>(dataset),
            cur_index_(0),
            reader_(dataset->env_, dataset->filename_) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(reader_.status());
        if (!reader_.Valid()) {
          return errors::Internal(
              "Cache iterator is in an invalid state. (Perhaps GetNext called "
              "after end_of_sequence?)");
        }
        out_tensors->clear();
        out_tensors->resize(dataset()->num_tensors_);

        for (size_t i = 0; i < dataset()->num_tensors_; ++i) {
          reader_.Next();  // The first entry in the table is a header entry.
          if (!reader_.Valid()) {
            out_tensors->clear();
            *end_of_sequence = true;
            return Status::OK();
          }
          StringPiece key = reader_.key();
          DCHECK_EQ(key, dataset()->FormatName(cur_index_, i));
          TF_RETURN_IF_ERROR(reader_.ReadCurrent(&(*out_tensors)[i]));
          TF_RETURN_IF_ERROR(reader_.status());
        }
        cur_index_++;
        return Status::OK();
      }

     private:
      mutex mu_;
      size_t cur_index_ GUARDED_BY(mu_);
      BundleReader reader_ GUARDED_BY(mu_);
    };  // FileReaderIterator

    const DatasetBase* const input_;
    const string filename_;
    Env* const env_;
    const size_t num_tensors_;
    const size_t tensor_index_padding_size_;
    static const size_t kMaxItems = 10000000;  // 10 million
    const size_t item_index_padding_size_;
    const string tensor_format_string_;
  };  // FileDataset

  class MemoryDataset : public DatasetBase {
   public:
    explicit MemoryDataset(const DatasetBase* input) : input_(input) {
      input->Ref();
    }

    ~MemoryDataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      mutex_lock l(mu_);
      if (cache_) {
        return std::unique_ptr<IteratorBase>(
            new MemoryReaderIterator(this, cache_.get()));
      }
      if (!writer_iterator_created_) {
        writer_iterator_created_ = true;
        return std::unique_ptr<IteratorBase>(new MemoryWriterIterator(this));
      }
      return std::unique_ptr<IteratorBase>(new DuplicateWriterIterator(this));
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() override { return "CacheDatasetOp::MemoryDataset"; }

   private:
    // MemoryWriterIterator passes through and appends items from the input
    // dataset to its vector.
    //
    // This iterator is used when dataset->cache_ is null. After buffering
    // the tensors in memory, upon exhausing the underlying iterator, they are
    // updated into the parent dataset's cache_ pointer.
    class MemoryWriterIterator : public DatasetIterator<MemoryDataset> {
     public:
      explicit MemoryWriterIterator(const MemoryDataset* dataset)
          : DatasetIterator<MemoryDataset>(dataset),
            input_impl_(dataset->input_->MakeIterator()),
            cache_(new std::vector<std::vector<Tensor>>) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence) {
          // Guard on cache_ to not crash if GetNext is called a second time
          // after *end_of_sequence == true
          if (cache_) {
            mutex_lock l2(dataset()->mu_);
            DCHECK(dataset()->writer_iterator_created_);
            DCHECK(!dataset()->cache_);
            cache_.swap(dataset()->cache_);
          }
          return Status::OK();
        }
        cache_->emplace_back(*out_tensors);
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::unique_ptr<std::vector<std::vector<Tensor>>> cache_ GUARDED_BY(mu_);
    };  // MemoryWriterIterator

    class MemoryReaderIterator : public DatasetIterator<MemoryDataset> {
     public:
      explicit MemoryReaderIterator(
          const MemoryDataset* dataset,
          const std::vector<std::vector<Tensor>>* cache)
          : DatasetIterator<MemoryDataset>(dataset), cache_(cache), index_(0) {
        CHECK(cache);
      }

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (index_ < cache_->size()) {
          const std::vector<Tensor>& cache_tensors = (*cache_)[index_];
          out_tensors->insert(out_tensors->begin(), cache_tensors.begin(),
                              cache_tensors.end());
          index_++;
          *end_of_sequence = false;
          return Status::OK();
        } else {
          *end_of_sequence = true;
          return Status::OK();
        }
      }

     private:
      mutex mu_;
      const std::vector<std::vector<Tensor>>* const cache_;
      size_t index_ GUARDED_BY(mu_);
    };  // MemoryReaderIterator

    class DuplicateWriterIterator : public DatasetIterator<MemoryDataset> {
     public:
      explicit DuplicateWriterIterator(const MemoryDataset* dataset)
          : DatasetIterator<MemoryDataset>(dataset) {}

      Status GetNext(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     bool* end_of_sequence) override {
        return errors::AlreadyExists(
            "There appears to be a concurrent caching iterator running.");
      }
    };  // DuplicateWriterIterator

    const DatasetBase* const input_;
    mutable mutex mu_;
    mutable std::unique_ptr<std::vector<std::vector<Tensor>>> cache_
        GUARDED_BY(mu_);
    mutable bool writer_iterator_created_ GUARDED_BY(mu_) = false;
  };  // MemoryDataset
};    // CacheDatasetOp

REGISTER_KERNEL_BUILDER(Name("CacheDataset").Device(DEVICE_CPU),
                        CacheDatasetOp);

}  // namespace

}  // namespace tensorflow
