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
#include <queue>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

class MatchingFilesDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* patterns_t;
    OP_REQUIRES_OK(ctx, ctx->input("patterns", &patterns_t));
    const auto patterns = patterns_t->flat<string>();
    size_t num_patterns = static_cast<size_t>(patterns.size());
    std::vector<string> pattern_strs;
    pattern_strs.reserve(num_patterns);

    for (int i = 0; i < num_patterns; ++i) {
      pattern_strs.push_back(patterns(i));
    }

    // keep the elements in the ascending order
    std::sort(pattern_strs.begin(), pattern_strs.end());
    *output = new Dataset(ctx, std::move(pattern_strs));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<string> patterns)
        : DatasetBase(DatasetContext(ctx)), patterns_(std::move(patterns)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::MatchingFiles")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override {
      return "MatchingFilesDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      Node* patterns_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(patterns_, &patterns_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {patterns_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        Status ret;

        while (!filepath_queue_.empty() ||
               current_pattern_index_ < dataset()->patterns_.size()) {
          // All the elements in the heap will be the matched filename or the
          // potential directory.
          if (!filepath_queue_.empty()) {
            string cur_file = filepath_queue_.top();
            filepath_queue_.pop();

            // We can also use isDectory() here. But IsDirectory call can be
            // expensive for some FS.
            if (ctx->env()->MatchPath(cur_file, current_pattern_)) {
              Tensor filepath_tensor(ctx->allocator({}), DT_STRING, {});
              filepath_tensor.scalar<string>()() = cur_file;
              out_tensors->emplace_back(std::move(filepath_tensor));
              *end_of_sequence = false;
              return Status::OK();
            }

            // In this case, cur_file is a directory. Then create a sub-pattern
            // to continue the search.
            size_t pos = current_pattern_.find_first_of("*?[\\");
            size_t len = current_pattern_.size() - pos;
            string cur_pattern_suffix = current_pattern_.substr(pos, len);
            string sub_pattern =
                strings::StrCat(cur_file, "/", cur_pattern_suffix);
            Status s = UpdateIterator(ctx, sub_pattern);
            ret.Update(s);
          } else {
            // search a new pattern
            current_pattern_ = dataset()->patterns_[current_pattern_index_];
            Status s = UpdateIterator(ctx, current_pattern_);
            ret.Update(s);
            ++current_pattern_index_;
          }
        }

        *end_of_sequence = true;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("current_pattern_index"), current_pattern_index_));

        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_pattern"),
                                               current_pattern_));

        if (!filepath_queue_.empty()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("queue_size"),
                                                 filepath_queue_.size()));
          for (int i = 0; i < filepath_queue_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("queue_element_", i)),
                filepath_queue_.top()));
            filepath_queue_.pop();
          }
        }
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        int64 current_pattern_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name("current_pattern_index"), &current_pattern_index));
        current_pattern_index_ = size_t(current_pattern_index);

        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_pattern"),
                                              &current_pattern_));

        int64 queue_size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("queue_size"), &queue_size));
        for (int i = static_cast<int>(queue_size - 1); i >= 0; --i) {
          string element;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              full_name(strings::StrCat("queue_element_", i)), &element));
          filepath_queue_.push(element);
        }
        return Status::OK();
      }

     private:
      Status UpdateIterator(IteratorContext* ctx, const string& pattern)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        string fixed_prefix = pattern.substr(0, pattern.find_first_of("*?[\\"));
        string eval_pattern = pattern;
        string dir(io::Dirname(fixed_prefix));

        // If dir is empty then we need to fix up fixed_prefix and eval_pattern
        // to include . as the top level directory.
        if (dir.empty()) {
          dir = ".";
          fixed_prefix = io::JoinPath(dir, fixed_prefix);
          eval_pattern = io::JoinPath(dir, pattern);
        }

        FileSystem* fs;
        TF_RETURN_IF_ERROR(ctx->env()->GetFileSystemForFile(dir, &fs));

        filepath_queue_.push(dir);
        Status ret;  // Status to return
        // children_dir_status holds is_dir status for children. It can have
        // three possible values: OK for true; FAILED_PRECONDITION for false;
        // CANCELLED if we don't calculate IsDirectory (we might do that because
        // there isn't any point in exploring that child path).

        // DFS to find the first element in the iterator.
        while (!filepath_queue_.empty()) {
          string cur_dir = filepath_queue_.top();
          filepath_queue_.pop();
          std::vector<string> children;
          Status s = fs->GetChildren(cur_dir, &children);
          ret.Update(s);

          // If cur_dir has no children, there will two possible situations: 1)
          // the cur_dir is an empty dir; 2) the cur_dir is actual a file
          // instead of a director. For the first one, continue to search the
          // heap. For the second one, if the file matches the pattern, add
          // it to the heap and finish the search; otherwise, continue the next
          // search.
          if (children.empty()) {
            if (ctx->env()->MatchPath(cur_dir, eval_pattern)) {
              filepath_queue_.push(cur_dir);
              return ret;
            } else {
              continue;
            }
          }

          std::map<string, Status> children_dir_status;
          // This IsDirectory call can be expensive for some FS. Parallelizing
          // it.
          ForEach(
              ctx, 0, children.size(),
              [fs, &cur_dir, &children, &fixed_prefix,
               &children_dir_status](int i) {
                const string child_path = io::JoinPath(cur_dir, children[i]);
                // In case the child_path doesn't start with the fixed_prefix,
                // then we don't need to explore this path.
                if (!str_util::StartsWith(child_path, fixed_prefix)) {
                  children_dir_status[child_path] = Status(
                      tensorflow::error::CANCELLED, "Operation not needed");
                } else {
                  children_dir_status[child_path] = fs->IsDirectory(child_path);
                }
              });

          for (const auto& child : children) {
            const string child_dir_path = io::JoinPath(cur_dir, child);
            const Status child_dir_status = children_dir_status[child];
            // If the IsDirectory call was cancelled we bail.
            if (child_dir_status.code() == tensorflow::error::CANCELLED) {
              continue;
            }

            if (child_dir_status.ok()) {
              // push the child dir for next search
              filepath_queue_.push(child_dir_path);
            } else {
              // This case will be a file: if the file matches the pattern, push
              // it to the heap; otherwise, ignore it.
              if (ctx->env()->MatchPath(child_dir_path, eval_pattern)) {
                filepath_queue_.push(child_dir_path);
              }
            }
          }
        }
        return ret;
      }

      static void ForEach(IteratorContext* ctx, int first, int last,
                          const std::function<void(int)>& f) {
        for (int i = first; i < last; i++) {
          (*ctx->runner())([f, i] { std::bind(f, i); });
        }
      }

      mutex mu_;
      std::priority_queue<string, std::vector<string>, std::less<string>>
          filepath_queue_ GUARDED_BY(mu_);
      size_t current_pattern_index_ GUARDED_BY(mu_) = 0;
      string current_pattern_ GUARDED_BY(mu_);
    };

    const std::vector<string> patterns_;
  };
};

REGISTER_KERNEL_BUILDER(Name("MatchingFilesDataset").Device(DEVICE_CPU),
                        MatchingFilesDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
