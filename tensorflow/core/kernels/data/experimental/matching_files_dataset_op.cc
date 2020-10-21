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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
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
namespace experimental {
namespace {

class MatchingFilesDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* patterns_t;
    OP_REQUIRES_OK(ctx, ctx->input("patterns", &patterns_t));
    const auto patterns = patterns_t->flat<tstring>();
    size_t num_patterns = static_cast<size_t>(patterns.size());
    std::vector<tstring> pattern_strs;
    pattern_strs.reserve(num_patterns);

    for (size_t i = 0; i < num_patterns; i++) {
      pattern_strs.push_back(patterns(i));
    }

    *output = new Dataset(ctx, std::move(pattern_strs));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<tstring> patterns)
        : DatasetBase(DatasetContext(ctx)), patterns_(std::move(patterns)) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::MatchingFiles")});
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

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
      return Status::OK();
    }

    Status CheckExternalState() const override { return Status::OK(); }

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
        FileSystem* fs;

        TF_RETURN_IF_ERROR(ctx->env()->GetFileSystemForFile(
            dataset()->patterns_[(current_pattern_index_ > 0)
                                     ? current_pattern_index_ - 1
                                     : 0],
            &fs));

        while (!filepath_queue_.empty() ||
               current_pattern_index_ < dataset()->patterns_.size()) {
          // All the elements in the heap will be the matched filenames or the
          // potential directories.
          if (!filepath_queue_.empty()) {
            PathStatus current_path = filepath_queue_.top();
            filepath_queue_.pop();

            if (!current_path.second) {
              Tensor filepath_tensor(ctx->allocator({}), DT_STRING, {});

              // Replace the forward slash with the backslash for Windows path
              if (isWindows_) {
                std::replace(current_path.first.begin(),
                             current_path.first.end(), '/', '\\');
              }

              filepath_tensor.scalar<tstring>()() =
                  std::move(current_path.first);
              out_tensors->emplace_back(std::move(filepath_tensor));
              *end_of_sequence = false;
              hasMatch_ = true;
              return Status::OK();
            }

            // In this case, current_path is a directory. Then continue the
            // search.
            TF_RETURN_IF_ERROR(
                UpdateIterator(ctx, fs, current_path.first, current_pattern_));
          } else {
            // search a new pattern
            current_pattern_ = dataset()->patterns_[current_pattern_index_];
            StringPiece current_pattern_view = StringPiece(current_pattern_);

            // Windows paths contain backslashes and Windows APIs accept forward
            // and backslashes equivalently, so we convert the pattern to use
            // forward slashes exclusively. The backslash is used as the
            // indicator of Windows paths. Note that this is not ideal, since
            // the API expects backslash as an escape character, but no code
            // appears to rely on this behavior
            if (current_pattern_view.find('\\') != std::string::npos) {
              isWindows_ = true;
              std::replace(&current_pattern_[0],
                           &current_pattern_[0] + current_pattern_.size(), '\\',
                           '/');
            } else {
              isWindows_ = false;
            }

            StringPiece fixed_prefix = current_pattern_view.substr(
                0, current_pattern_view.find_first_of("*?[\\"));
            string current_dir(io::Dirname(fixed_prefix));

            // If current_dir is empty then we need to fix up fixed_prefix and
            // current_pattern_ to include . as the top level directory.
            if (current_dir.empty()) {
              current_dir = ".";
              current_pattern_ = io::JoinPath(current_dir, current_pattern_);
            }

            TF_RETURN_IF_ERROR(
                UpdateIterator(ctx, fs, current_dir, current_pattern_));
            ++current_pattern_index_;
          }
        }

        *end_of_sequence = true;
        if (hasMatch_) {
          return Status::OK();
        } else {
          return errors::NotFound("Don't find any matched files");
        }
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeSourceNode(std::move(args));
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name("current_pattern_index"), current_pattern_index_));

        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_pattern"),
                                               current_pattern_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("hasMatch"), hasMatch_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("isWindows"), isWindows_));

        if (!filepath_queue_.empty()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("queue_size"),
                                                 filepath_queue_.size()));
          int i = 0;
          while (!filepath_queue_.empty()) {
            TF_RETURN_IF_ERROR(
                writer->WriteScalar(full_name(strings::StrCat("path_", i)),
                                    filepath_queue_.top().first));
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("path_status_", i)),
                filepath_queue_.top().second));
            filepath_queue_.pop();
            i++;
          }
        }

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        int64 current_pattern_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name("current_pattern_index"), &current_pattern_index));
        current_pattern_index_ = size_t(current_pattern_index);

        tstring current_pattern_tstr;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_pattern"),
                                              &current_pattern_tstr));
        current_pattern_ = current_pattern_tstr;

        int64 hasMatch;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("hasMatch"), &hasMatch));
        hasMatch_ = static_cast<bool>(hasMatch);

        int64 isWindows;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("isWindows"), &isWindows));
        isWindows_ = static_cast<bool>(isWindows);

        if (reader->Contains(full_name("queue_size"))) {
          int64 queue_size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("queue_size"), &queue_size));
          for (int i = 0; i < queue_size; i++) {
            tstring path;
            int64 path_status;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("path_", i)), &path));
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("path_status_", i)), &path_status));
            filepath_queue_.push(
                PathStatus(path, static_cast<bool>(path_status)));
          }
        }

        return Status::OK();
      }

     private:
      Status UpdateIterator(IteratorContext* ctx, FileSystem* fs,
                            const string& dir, const string& eval_pattern)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        StringPiece fixed_prefix =
            StringPiece(eval_pattern)
                .substr(0, eval_pattern.find_first_of("*?[\\"));

        filepath_queue_.push(PathStatus(dir, true));
        Status ret;  // Status to return

        // DFS to find the first element in the iterator.
        while (!filepath_queue_.empty()) {
          const PathStatus current_path = filepath_queue_.top();

          // All the files in the heap are matched with the pattern, so finish
          // the search if current_path is a file.
          if (!current_path.second) {
            return Status::OK();
          }

          filepath_queue_.pop();

          // If current_path is a directory, search its children.
          const string& current_dir = current_path.first;
          std::vector<string> children;
          ret.Update(fs->GetChildren(current_dir, &children));

          // Handle the error cases: 1) continue the search if the status is
          // NOT_FOUND; 2) return the non-ok status immediately if it is not
          // NOT_FOUND.
          if (ret.code() == error::NOT_FOUND) {
            continue;
          } else if (!ret.ok()) {
            return ret;
          }

          // children_dir_status holds is_dir status for children. It can have
          // three possible values: OK for true; FAILED_PRECONDITION for false;
          // CANCELLED if we don't calculate IsDirectory (we might do that
          // because there isn't any point in exploring that child path).
          std::vector<Status> children_dir_status;
          children_dir_status.resize(children.size());

          // This IsDirectory call can be expensive for some FS. Parallelizing
          // it.
          auto is_directory_fn = [fs, current_dir, &children, &fixed_prefix,
                                  &children_dir_status](int i) {
            const string child_path = io::JoinPath(current_dir, children[i]);
            // In case the child_path doesn't start with the fixed_prefix, then
            // we don't need to explore this path.
            if (!absl::StartsWith(child_path, fixed_prefix)) {
              children_dir_status[i] =
                  errors::Cancelled("Operation not needed");
            } else {
              children_dir_status[i] = fs->IsDirectory(child_path);
            }
          };

          BlockingCounter counter(children.size());
          for (int i = 0; i < children.size(); i++) {
            (*ctx->runner())([&is_directory_fn, &counter, i] {
              is_directory_fn(i);
              counter.DecrementCount();
            });
          }
          counter.Wait();

          for (int i = 0; i < children.size(); i++) {
            const string& child_dir_path =
                io::JoinPath(current_dir, children[i]);
            const Status& child_dir_status = children_dir_status[i];

            // If the IsDirectory call was cancelled we bail.
            if (child_dir_status.code() == tensorflow::error::CANCELLED) {
              continue;
            }

            if (child_dir_status.ok()) {
              // push the child dir for next search
              filepath_queue_.push(PathStatus(child_dir_path, true));
            } else {
              // This case will be a file: if the file matches the pattern, push
              // it to the heap; otherwise, ignore it.
              if (ctx->env()->MatchPath(child_dir_path, eval_pattern)) {
                filepath_queue_.push(PathStatus(child_dir_path, false));
              }
            }
          }
        }
        return ret;
      }

      mutex mu_;
      // True means the path is a directory; False means the path is a filename.
      typedef std::pair<string, bool> PathStatus;
      std::priority_queue<PathStatus, std::vector<PathStatus>,
                          std::greater<PathStatus>>
          filepath_queue_ TF_GUARDED_BY(mu_);
      size_t current_pattern_index_ TF_GUARDED_BY(mu_) = 0;
      tstring current_pattern_ TF_GUARDED_BY(mu_);
      bool hasMatch_ TF_GUARDED_BY(mu_) = false;
      bool isWindows_ TF_GUARDED_BY(mu_) = false;
    };

    const std::vector<tstring> patterns_;
  };
};

REGISTER_KERNEL_BUILDER(Name("MatchingFilesDataset").Device(DEVICE_CPU),
                        MatchingFilesDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalMatchingFilesDataset").Device(DEVICE_CPU),
    MatchingFilesDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
