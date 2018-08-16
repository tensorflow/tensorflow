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
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

class WindowDataset : public DatasetBase {
 public:
  WindowDataset(std::vector<std::vector<Tensor>> elements,
                DataTypeVector output_types,
                std::vector<PartialTensorShape> output_shapes)
      : DatasetBase(DatasetContext({"Window"})),
        elements_(std::move(elements)),
        output_types_(std::move(output_types)),
        output_shapes_(std::move(output_shapes)) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::unique_ptr<IteratorBase>(
        new Iterator({this, strings::StrCat(prefix, "::Window")}));
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override { return "WindowDataset"; }

 protected:
  // TODO(b/110981596): Support checkpointing.
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    return errors::Unimplemented("%s does not support serialization",
                                 DebugString());
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

    Status SaveInternal(IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("i"), i_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      int64 i;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("i"), &i));
      i_ = size_t(i);
      return Status::OK();
    }

    mutex mu_;
    size_t i_ GUARDED_BY(mu_) = 0;
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

}  // namespace tensorflow
