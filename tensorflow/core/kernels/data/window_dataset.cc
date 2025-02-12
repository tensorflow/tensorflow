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

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kInputs[] = "inputs";
constexpr char kOutputTypes[] = "output_types";
constexpr char kOutputShapes[] = "output_shapes";
constexpr char kWindow[] = "Window";
constexpr char kWindowOp[] = "WindowOp";
constexpr char kCurIndex[] = "i";

class Window : public DatasetBase {
 public:
  Window(std::vector<std::vector<Tensor>> elements, DataTypeVector output_types,
         std::vector<PartialTensorShape> output_shapes)
      : DatasetBase(DatasetContext({kWindowOp, kWindow})),
        elements_(std::move(elements)),
        output_types_(std::move(output_types)),
        output_shapes_(std::move(output_shapes)) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::make_unique<Iterator>(
        Iterator::Params{this, name_utils::IteratorPrefix(kWindow, prefix)});
  }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  int64_t AllocatedBytes() const override {
    int64_t allocated_bytes = 0;
    for (auto& element : elements_) {
      allocated_bytes += GetAllocatedBytes(element);
    }
    return allocated_bytes;
  }

  int64_t TotalBytes() const override {
    int64_t total_bytes = 0;
    for (auto& element : elements_) {
      total_bytes += GetTotalBytes(element);
    }
    return total_bytes;
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    return elements_.size();
  }

  string DebugString() const override { return kWindow; }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override { return absl::OkStatus(); }

 protected:
  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    if (ctx->is_graph_rewrite()) {
      // If data tensors are not to be serialized (e.g. when the serialization
      // is done for the sake of graph optimizations), we return
      // `errors::Unimplemented` to short-circuit the computation.
      return errors::Unimplemented(DebugString(),
                                   " does not support serialization");
    }
    std::vector<Node*> input_nodes;
    for (const auto& element : elements_) {
      for (const auto& t : element) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddDatasetOrTensor(ctx, t, &node));
        input_nodes.emplace_back(node);
      }
    }
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {}, {std::make_pair(0, input_nodes)}, {}, output));
    return absl::OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Window> {
   public:
    explicit Iterator(const Params& params) : DatasetIterator<Window>(params) {}

    absl::Status GetNextInternal(IteratorContext* ctx,
                                 std::vector<Tensor>* out_tensors,
                                 bool* end_of_sequence) override {
      mutex_lock l(mu_);
      if (i_ == dataset()->elements_.size()) {
        *end_of_sequence = true;
      } else {
        *end_of_sequence = false;
        *out_tensors = dataset()->elements_[i_++];
      }
      return absl::OkStatus();
    }

    absl::Status SaveInternal(SerializationContext* ctx,
                              IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix(), kCurIndex, i_));
      return absl::OkStatus();
    }

    absl::Status RestoreInternal(IteratorContext* ctx,
                                 IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      int64_t i;
      TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kCurIndex, &i));
      i_ = size_t(i);
      return absl::OkStatus();
    }

    mutex mu_;
    size_t i_ TF_GUARDED_BY(mu_) = 0;
  };

  const std::vector<std::vector<Tensor>> elements_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

class WindowOp : public DatasetOpKernel {
 public:
  explicit WindowOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list(kInputs, &inputs));
    auto element_size = output_shapes_.size();
    auto num_elements = ctx->num_inputs() / element_size;
    std::vector<std::vector<Tensor>> elements;
    for (size_t i = 0; i < num_elements; ++i) {
      std::vector<Tensor> element;
      element.reserve(element_size);
      for (size_t j = 0; j < element_size; ++j) {
        element.push_back(std::move(inputs[i * element_size + j]));
      }
      elements.push_back(std::move(element));
    }
    *output = new Window(std::move(elements), output_types_, output_shapes_);
  }

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("WindowOp").Device(DEVICE_CPU), WindowOp);

}  // namespace

absl::Status NewWindow(std::vector<std::vector<Tensor>> elements,
                       DataTypeVector output_types,
                       std::vector<PartialTensorShape> output_shapes,
                       DatasetBase** out_dataset) {
  // TODO(mrry): If this becomes more public, we must validate that
  // the elements match the output_types and output_shapes.
  *out_dataset = new Window(std::move(elements), std::move(output_types),
                            std::move(output_shapes));
  (*out_dataset)->Initialize(/*metadata=*/{});
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
