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

class CacheDatasetOp : public OpKernel {
 public:
  explicit CacheDatasetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &input));
    core::ScopedUnref unref_input(input);

    // Parse out the filenames tensor.
    const Tensor* filename_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("filename", &filename_tensor));
    OP_REQUIRES(ctx, filename_tensor->dims() == 0,
                errors::InvalidArgument("`filename` must be a scalar."));
    string filename = filename_tensor->flat<string>()(0);

    DatasetBase* dataset = new Dataset(input, filename, ctx->env());
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    ResourceHandle handle = MakeResourceHandle<DatasetBase>(
        ctx, ctx->step_container()->name(), name());
    OP_REQUIRES_OK(ctx, CreateResource(ctx, handle, dataset));
    output->flat<ResourceHandle>()(0) = handle;
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(const DatasetBase* input, string filename, Env* env)
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

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIterator() const override {
      if (env_->FileExists(strings::StrCat(filename_, ".index")).ok()) {
        return std::unique_ptr<IteratorBase>(new ReaderIterator(this));
      } else {
        return std::unique_ptr<IteratorBase>(new WriterIterator(this));
      }
    }

    const DataTypeVector& output_dtypes() const override {
      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return input_->output_shapes();
    }

    string DebugString() override { return "CacheDatasetOp::Dataset"; }

   private:
    static size_t StringPaddingSize(size_t num_tensors) {
      return strings::Printf("%zu", num_tensors - 1).size();
    }

    string FormatName(size_t item_index, size_t tensor_index) const {
      return strings::Printf(tensor_format_string_.c_str(), item_index,
                             tensor_index);
    }

    // WriterIterator passes through and caches items from the input dataset.
    //
    // This iterator is used when the cache directory is not found on disk. It
    // creates the cache directory, and passes on the underlying iterator's
    // elements.
    class WriterIterator : public DatasetIterator<Dataset> {
     public:
      explicit WriterIterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
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
    };  // WriterIterator

    class ReaderIterator : public DatasetIterator<Dataset> {
     public:
      explicit ReaderIterator(const Dataset* dataset)
          : DatasetIterator<Dataset>(dataset),
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
    };  // ReaderIterator

    const DatasetBase* const input_;
    const string filename_;
    Env* const env_;
    const size_t num_tensors_;
    const size_t tensor_index_padding_size_;
    static const size_t kMaxItems = 10000000;  // 10 million
    const size_t item_index_padding_size_;
    const string tensor_format_string_;
  };  // Dataset
};    // CacheDatasetOp

REGISTER_KERNEL_BUILDER(Name("CacheDataset").Device(DEVICE_CPU),
                        CacheDatasetOp);

}  // namespace

}  // namespace tensorflow
