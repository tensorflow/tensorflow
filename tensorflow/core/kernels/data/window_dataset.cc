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
#include "tensorflow/core/kernels/data/window_dataset.h"

#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kWindow[] = "Window";
constexpr char kWindowDataset[] = "WindowDataset";
constexpr char kCurIndex[] = "i";

class WindowDataset : public DatasetBase {
 public:
  WindowDataset(std::vector<std::vector<Tensor>> elements,
                DataTypeVector output_types,
                std::vector<PartialTensorShape> output_shapes)
      : DatasetBase(DatasetContext({kWindow})),
        elements_(std::move(elements)),
        output_types_(std::move(output_types)),
        output_shapes_(std::move(output_shapes)) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, name_utils::IteratorPrefix(kWindow, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  int64 AllocatedBytes() const override {
    int64 allocated_bytes = 0;
    for (auto& element : elements_) {
      allocated_bytes += GetAllocatedBytes(element);
    }
    return allocated_bytes;
  }

  int64 TotalBytes() const override {
    int64 total_bytes = 0;
    for (auto& element : elements_) {
      total_bytes += GetTotalBytes(element);
    }
    return total_bytes;
  }

  int64 Cardinality() const override { return elements_.size(); }

  string DebugString() const override { return kWindowDataset; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    return Status::OK();
  }

  Status CheckExternalState() const override { return Status::OK(); }

 protected:
  // TODO(b/110981596): Support checkpointing.
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return errors::Unimplemented(DebugString(),
                                 " does not support serialization");
  }

 private:
  class Iterator : public DatasetIterator<WindowDataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<WindowDataset>(params) {}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (i_ == dataset()->elements_.size()) {
        *end_of_sequence = true;
      } else {
        *end_of_sequence = false;
        *out_tensors = dataset()->elements_[i_++];
      }
      return Status::OK();
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurIndex), i_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      int64 i;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIndex), &i));
      i_ = size_t(i);
      return Status::OK();
    }

    mutex mu_;
    size_t i_ TF_GUARDED_BY(mu_) = 0;
  };

  const std::vector<std::vector<Tensor>> elements_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace

Status NewWindowDataset(std::vector<std::vector<Tensor>> elements,
                        DataTypeVector output_types,
                        std::vector<PartialTensorShape> output_shapes,
                        DatasetBase** out_dataset) {
  // TODO(mrry): If this becomes more public, we must validate that
  // the elements match the output_types and output_shapes.
  *out_dataset = new WindowDataset(std::move(elements), std::move(output_types),
                                   std::move(output_shapes));
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
