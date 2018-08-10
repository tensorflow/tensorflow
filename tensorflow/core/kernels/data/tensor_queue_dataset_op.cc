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

#include <deque>

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {

namespace {

bool IsGreaterEqualToOrCompatibleWith(const PartialTensorShape& a,
                                      const PartialTensorShape& b) {
  // Returns true if dims[a] >= dims[b], or are compatible.
  if (a.unknown_rank()) return true;
  if (a.dims() != b.dims()) return false;
  for (int d = 0; d < a.dims(); ++d) {
    if (a.dim_size(d) == -1 || b.dim_size(d) == -1) continue;
    if (a.dim_size(d) < b.dim_size(d)) return false;
  }
  return true;
}

DataTypeVector PrependQueueType(const DataTypeVector& dtypes) {
  DataTypeVector out;
  out.reserve(dtypes.size() + 1);
  out.push_back(DT_VARIANT);  // The queue component.
  for (const DataType& d : dtypes) out.push_back(d);
  return out;
}

std::vector<PartialTensorShape> PrependQueueShapeWithBatch(
    const std::vector<PartialTensorShape>& shapes) {
  std::vector<PartialTensorShape> out;
  out.reserve(shapes.size() + 1);
  out.emplace_back(PartialTensorShape({-1}));  // The queue component.
  for (PartialTensorShape s : shapes) {
    s.InsertDim(0, -1);  // Unknown batch size.
    out.push_back(std::move(s));
  }
  return out;
}

class EnqueueInQueueDatasetOp;

class PrependFromQueueAndPaddedBatchDataset : public GraphDatasetBase {
 public:
  PrependFromQueueAndPaddedBatchDataset(
      OpKernelContext* ctx, const int64 batch_size, const DatasetBase* input,
      const DataTypeVector& dtypes,
      const std::vector<PartialTensorShape>& shapes,
      std::vector<Tensor> padding_values)
      : GraphDatasetBase(ctx),
        batch_size_(batch_size),
        input_(input),
        dtypes_(dtypes),
        shapes_(shapes),
        padding_values_(std::move(padding_values)),
        dtypes_with_queue_(PrependQueueType(dtypes)),
        batched_shapes_with_queue_(PrependQueueShapeWithBatch(shapes)) {
    input_->Ref();
  }

  ~PrependFromQueueAndPaddedBatchDataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return std::unique_ptr<IteratorBase>(new Iterator(
        {this, strings::StrCat(prefix, "::PrependFromQueueAndPaddedBatch")}));
  }

  const DataTypeVector& output_dtypes() const override {
    return dtypes_with_queue_;
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
    return batched_shapes_with_queue_;
  }

  string DebugString() const override {
    return "PrependFromQueueAndPaddedBatchDatasetOp::Dataset";
  }

 protected:
  Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph = nullptr;
    TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input_, &input_graph));
    Node* batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));

    std::vector<Node*> padded_shapes;
    padded_shapes.reserve(shapes_.size());
    for (int i = 0; i < shapes_.size(); i++) {
      Node* node;
      Tensor t(DT_INT64, TensorShape({shapes_[i].dims()}));
      for (int j = 0; j < shapes_[i].dims(); j++) {
        t.vec<int64>()(j) = shapes_[i].dim_size(j);
      }
      TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
      padded_shapes.emplace_back(node);
    }

    std::vector<Node*> padding_values;
    padding_values.reserve(padding_values_.size());
    for (const Tensor& t : padding_values_) {
      Node* node;
      TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
      padding_values.emplace_back(node);
    }

    AttrValue output_types;
    b->BuildAttrValue(dtypes_, &output_types);

    AttrValue output_shapes;
    b->BuildAttrValue(batched_shapes_with_queue_, &output_shapes);

    AttrValue N;
    b->BuildAttrValue<int64>(shapes_.size(), &N);

    TF_RETURN_IF_ERROR(b->AddDataset(this, {{0, input_graph}, {1, batch_size}},
                                     {{2, padded_shapes}, {3, padding_values}},
                                     {{"Toutput_types", output_types},
                                      {"output_shapes", output_shapes},
                                      {"N", N}},
                                     output));

    return Status::OK();
  }

 private:
  friend class EnqueueInQueueDatasetOp;

  class Iterator
      : public DatasetIterator<PrependFromQueueAndPaddedBatchDataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<PrependFromQueueAndPaddedBatchDataset>(params) {}

    ~Iterator() override { queue_->Unref(); }

    Status Initialize(IteratorContext* ctx) override {
      std::unique_ptr<IteratorBase> iterator;
      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, prefix(), &iterator));
      queue_ = new TensorQueue(std::move(iterator), dataset()->dtypes_,
                               dataset()->shapes_);
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      std::vector<std::vector<Tensor>> batch;
      TF_RETURN_IF_ERROR(queue_->GetNext(ctx, dataset()->batch_size_, &batch,
                                         end_of_sequence));
      const auto& dtypes = dataset()->dtypes_;
      const auto& shapes = dataset()->shapes_;
      const auto& input_shapes = dataset()->input_->output_shapes();
      const auto& padding_values = dataset()->padding_values_;
      const int64 batch_size = batch.size();
      out_tensors->reserve(dtypes.size());

      std::vector<TensorShape> max_shapes;  // Of non-queue components.
      for (int i = 0; i < dtypes.size(); ++i) {
        const PartialTensorShape& shape = shapes[i];
        TensorShape out_shape({batch_size});
        for (int r = 0; r < shape.dims(); ++r) {
          if (shape.dim_size(r) >= 0) {
            // padded_shape[r] is known.
            out_shape.AddDim(shape.dim_size(r));
          } else {
            // padded_shape[r] is unknown, find the maximum across
            // the batch.
            int64 dim = 0;
            for (int b = 0; b < batch.size(); ++b) {
              dim = std::max(dim, batch[b][i].dim_size(r));
            }
            out_shape.AddDim(dim);
          }
        }
        max_shapes.push_back(std::move(out_shape));
      }

      Tensor queues_t(cpu_allocator(), DT_VARIANT, TensorShape({batch_size}));
      if (!batch.empty()) {
        auto queues = queues_t.flat<Variant>();
        Variant& queue_inserter = queues(0);
        queue_inserter = TensorQueueInserter();
        queue_inserter.get<TensorQueueInserter>()->set_queue(queue_);
        for (int b = 1; b < batch.size(); ++b) {
          // Copy the TensorQueueInserter.  Each copy increments the
          // Ref on the queue_.
          queues(b) = queues(0);
        }
      }
      out_tensors->push_back(std::move(queues_t));

      for (int i = 0; i < max_shapes.size(); ++i) {
        Tensor component(cpu_allocator(), dtypes[i], max_shapes[i]);
        // Try hard to take the fast path.
        if (shapes[i].IsFullyDefined() &&
            shapes[i].IsIdenticalTo(input_shapes[i])) {
          // Take the fast path if we know all the shapes statically.
          for (int64 b = 0; b < batch.size(); ++b) {
            TF_RETURN_IF_ERROR(
                batch_util::CopyElementToSlice(batch[b][i], &component, b));
          }
        } else {
          TF_RETURN_IF_ERROR(
              batch_util::SetElementZero(&component, padding_values[i]));
          for (int64 b = 0; b < batch.size(); ++b) {
            if (batch[b][i].shape() == max_shapes[i]) {
              TF_RETURN_IF_ERROR(
                  batch_util::CopyElementToSlice(batch[b][i], &component, b));
            } else {
              TF_RETURN_IF_ERROR(batch_util::CopyElementToLargerSlice(
                  batch[b][i], &component, b));
            }
          }
        }
        out_tensors->push_back(std::move(component));
      }

      // end_of_sequence was set before we populated out_tensors, so
      // it's ok to return now.
      return Status::OK();
    }

   protected:
    // Work around bug in MSVC that disallows access to protected
    // members of Iterator from within TensorQueue.
    class TensorQueue;
    friend class TensorQueue;

    class TensorQueue : public core::RefCounted {
     public:
      TensorQueue(std::unique_ptr<IteratorBase> input_impl,
                  const DataTypeVector& dtypes,
                  const std::vector<PartialTensorShape>& shapes)
          : dtypes_(dtypes),
            shapes_(shapes),
            input_impl_(std::move(input_impl)) {}

      void MaybeWaitForNotificationLocked(mutex_lock* lock)
          EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        // This essentially just releases the lock and immediately relocks.
        cv_.wait_for(*lock, std::chrono::milliseconds(0));
      }

      void NotifyLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) { cv_.notify_all(); }

      Status GetNext(IteratorContext* ctx, const int64 batch_size,
                     std::vector<std::vector<Tensor>>* batch,
                     bool* end_of_sequence) {
        mutex_lock lock(mu_);

        *end_of_sequence = false;

        for (int64 b = 0; b < batch_size;) {
          if (!entries_.empty()) {
            batch->push_back(std::move(entries_.front()));
            entries_.pop_front();
            ++b;
            continue;
          } else {
            if (input_impl_) {
              // There's still input coming in.
              std::vector<Tensor> tensors;
              bool input_end;
              TF_RETURN_IF_ERROR(
                  input_impl_->GetNext(ctx, &tensors, &input_end));
              if (!input_end) {
                batch->push_back(std::move(tensors));
                ++b;
                continue;
              } else {
                input_impl_.reset();
              }
            }
            if (!input_impl_) {
              // There's no more input coming in.
              if (RefCountIsOne()) {
                // No TensorQueueInserters in the wild.
                if (batch->empty()) {
                  *end_of_sequence = true;
                }
                break;
              } else {
                MaybeWaitForNotificationLocked(&lock);
                // If there's data available, try to add entries again.
                // Otherwise return a smaller batch and hope the next
                // iterator request has a non-empty or unused queue_.
                if (entries_.empty()) {
                  break;
                }
              }
            }
          }
        }  // for (int64 b = ... batch_size)
        return Status::OK();
      }

      Status Insert(const std::vector<Tensor>& tensors) {
        if (tensors.size() != dtypes_.size()) {
          return errors::InvalidArgument(
              "TensorQueue::Insert: mismatched number of tensors.  Queue "
              "expects ",
              dtypes_.size(), " tensors but tried to insert ", tensors.size());
        }
        for (int i = 0; i < tensors.size(); ++i) {
          if (tensors[i].dtype() != dtypes_[i]) {
            return errors::InvalidArgument(
                "TensorQueue::Insert: mismatched dtypes at component ", i,
                ".  Attempted "
                "to insert tensor of type ",
                DataTypeString(tensors[i].dtype()),
                " but queue expected type: ", DataTypeString(dtypes_[i]));
          }
          if (!shapes_[i].IsCompatibleWith(tensors[i].shape())) {
            return errors::InvalidArgument(
                "TensorQueue::Insert: mismatched shapes at component ", i,
                ".  Attempted "
                "to insert tensor with shape ",
                tensors[i].shape().DebugString(),
                " but queue expected shape: ", shapes_[i].DebugString());
          }
        }
        mutex_lock lock(mu_);
        entries_.push_back(tensors);
        NotifyLocked();
        return Status::OK();
      }

      Status Save(Iterator* iter, IteratorStateWriter* writer) {
        mutex_lock lock(mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(iter->SaveParent(writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(iter->full_name("input_exhausted"), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(iter->full_name("entries_size"),
                                               entries_.size()));
        for (int64 b = 0; b < entries_.size(); ++b) {
          for (int i = 0; i < dtypes_.size(); ++i) {
            TF_RETURN_IF_ERROR(
                writer->WriteTensor(strings::StrCat(iter->full_name("entries"),
                                                    "[", b, "][", i, "]"),
                                    entries_[b][i]));
          }
        }
        return Status::OK();
      }

      Status Restore(Iterator* iter, IteratorContext* ctx,
                     IteratorStateReader* reader) {
        mutex_lock l(mu_);
        if (reader->Contains(iter->full_name("input_exhausted"))) {
          input_impl_.reset();
        } else {
          TF_RETURN_IF_ERROR(iter->dataset_input()->MakeIterator(
              ctx, iter->prefix(), &input_impl_));
          TF_RETURN_IF_ERROR(iter->RestoreParent(ctx, reader, input_impl_));
        }
        entries_.clear();
        int64 entries_size = -1;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(iter->full_name("entries_size"), &entries_size));
        if (entries_size < 0) {
          return errors::DataLoss(
              "Expected entries_size key '", iter->full_name("entries_size"),
              "' to have nonnegative value, but saw: ", entries_size);
        }
        for (int64 b = 0; b < entries_size; ++b) {
          std::vector<Tensor> entry;
          for (int i = 0; i < dtypes_.size(); ++i) {
            Tensor value;
            TF_RETURN_IF_ERROR(
                reader->ReadTensor(strings::StrCat(iter->full_name("entries"),
                                                   "[", b, "][", i, "]"),
                                   &value));
            entry.push_back(std::move(value));
          }
          entries_.push_back(std::move(entry));
        }
        return Status::OK();
      }

      mutex* mu() { return &mu_; }

     private:
      DataTypeVector dtypes_;
      std::vector<PartialTensorShape> shapes_;

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      std::deque<std::vector<Tensor>> entries_ GUARDED_BY(mu_);
      condition_variable cv_ GUARDED_BY(mu_);
    };

    const DatasetBase* dataset_input() const { return dataset()->input_; }

    Status SaveInternal(IteratorStateWriter* writer) override {
      return queue_->Save(this, writer);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      return queue_->Restore(this, ctx, reader);
    }

   public:
    class TensorQueueInserter {
     public:
      TensorQueueInserter() : queue_(nullptr) {}

      void set_queue(TensorQueue* queue) {
        queue_ = queue;
        queue_->Ref();
      }

      TensorQueueInserter(const TensorQueueInserter& rhs) {
        queue_ = rhs.queue_;
        queue_->Ref();
      };

      TensorQueueInserter(TensorQueueInserter&& rhs) {
        queue_ = rhs.queue_;
        rhs.queue_ = nullptr;
      }

      TensorQueueInserter& operator=(const TensorQueueInserter& rhs) = delete;

      string TypeName() const { return "tensorflow::TensorQueueInserter"; }
      string DebugString() const { return TypeName(); }

      void Encode(VariantTensorData*) const {}
      bool Decode(const VariantTensorData&) { return false; }

      ~TensorQueueInserter() {
        if (queue_) {
          mutex_lock lock(*queue_->mu());
          queue_->Unref();
          queue_->NotifyLocked();
          queue_ = nullptr;
        }
      }

      Status Insert(const std::vector<Tensor>& tensors) const {
        CHECK(queue_);
        return queue_->Insert(tensors);
      }

     private:
      mutable TensorQueue* queue_;
    };

   private:
    TensorQueue* queue_;
  };

 private:
  const int64 batch_size_;
  const DatasetBase* input_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
  const std::vector<Tensor> padding_values_;
  const DataTypeVector dtypes_with_queue_;
  const std::vector<PartialTensorShape> batched_shapes_with_queue_;
};

class PrependFromQueueAndPaddedBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit PrependFromQueueAndPaddedBatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Toutput_types", &output_types_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    int64 batch_size = 0;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "batch_size", &batch_size));
    OP_REQUIRES(
        ctx, batch_size > 0,
        errors::InvalidArgument("Batch size must be greater than zero."));

    OpInputList padded_shape_tensors;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("padded_shapes", &padded_shape_tensors));
    std::vector<PartialTensorShape> padded_shapes;
    padded_shapes.reserve(padded_shape_tensors.size());
    OP_REQUIRES(ctx,
                padded_shape_tensors.size() == input->output_shapes().size(),
                errors::InvalidArgument("Number of padded shapes (",
                                        padded_shape_tensors.size(),
                                        ") must match the number of components "
                                        "in the input dataset's elements (",
                                        input->output_shapes().size(), ")"));
    for (const Tensor& padded_shape_t : padded_shape_tensors) {
      OP_REQUIRES(ctx, TensorShapeUtils::IsVector(padded_shape_t.shape()),
                  errors::InvalidArgument("All padded shapes must be vectors"));
      PartialTensorShape padded_shape;
      OP_REQUIRES_OK(ctx, PartialTensorShape::MakePartialShape(
                              padded_shape_t.vec<int64>().data(),
                              padded_shape_t.NumElements(), &padded_shape));
      padded_shapes.push_back(std::move(padded_shape));
    }

    OP_REQUIRES(
        ctx, input->output_dtypes() == output_types_,
        errors::InvalidArgument("Input dataset and this dataset "
                                "have different output_types: ",
                                DataTypeVectorString(input->output_dtypes()),
                                " and ", DataTypeVectorString(output_types_)));

    for (int i = 0; i < input->output_shapes().size(); ++i) {
      // Exclude the queue from the tensor_shapes calculation.
      const PartialTensorShape& tensor_shape = padded_shapes[i];
      OP_REQUIRES(
          ctx,
          IsGreaterEqualToOrCompatibleWith(tensor_shape,
                                           input->output_shapes()[i]),
          errors::InvalidArgument("Incompatible input shapes at component ", i,
                                  " between input dataset this dataset: ",
                                  input->output_shapes()[i].DebugString(),
                                  " vs. ", tensor_shape.DebugString()));
    }

    OpInputList padding_values_list;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("padding_values", &padding_values_list));
    std::vector<Tensor> padding_values;
    OP_REQUIRES(ctx,
                padding_values_list.size() == input->output_shapes().size(),
                errors::InvalidArgument(
                    "Number of padding values (", padding_values_list.size(),
                    ") must match the number of components in the input "
                    "dataset's elements (",
                    input->output_shapes().size(), ")"));
    for (int i = 0; i < padding_values_list.size(); ++i) {
      const Tensor& padding_value_t = padding_values_list[i];
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(padding_value_t.shape()),
          errors::InvalidArgument(
              "All padding values must be scalars; but at component ", i,
              " saw shape: ", padding_value_t.shape().DebugString()));
      OP_REQUIRES(ctx, padding_value_t.dtype() == input->output_dtypes()[i],
                  errors::InvalidArgument(
                      "Mismatched type between padding value ", i,
                      " and input dataset's component ", i, ": ",
                      DataTypeString(padding_value_t.dtype()), " vs. ",
                      DataTypeString(input->output_dtypes()[i])));
      padding_values.push_back(padding_value_t);
    }

    *output = new PrependFromQueueAndPaddedBatchDataset(
        ctx, batch_size, input, output_types_, padded_shapes,
        std::move(padding_values));
  }

 private:
  DataTypeVector output_types_;
};

REGISTER_KERNEL_BUILDER(
    Name("PrependFromQueueAndPaddedBatchDataset").Device(DEVICE_CPU),
    PrependFromQueueAndPaddedBatchDatasetOp);

class EnqueueInQueueDatasetOp : public OpKernel {
 public:
  explicit EnqueueInQueueDatasetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    using TensorQueueInserter =
        PrependFromQueueAndPaddedBatchDataset::Iterator::TensorQueueInserter;

    // TODO(ebrevdo): accept list of sequence lengths to do proper
    // sub-slicing of tensors for placement into the queue?
    const Tensor& tensor_queue_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(tensor_queue_t.shape()),
                errors::InvalidArgument("queue must be a vector, saw shape: ",
                                        tensor_queue_t.shape().DebugString()));
    std::vector<const TensorQueueInserter*> inserters;
    const int64 batch_size = tensor_queue_t.NumElements();
    inserters.reserve(batch_size);
    const Variant* variants = tensor_queue_t.flat<Variant>().data();
    for (int i = 0; i < batch_size; ++i) {
      const auto* inserter = variants[i].get<TensorQueueInserter>();
      OP_REQUIRES(ctx, inserter != nullptr,
                  errors::InvalidArgument(
                      "Could not access TensorQueueInserter from queue[", i,
                      "].  Received variant: ", variants[i].DebugString()));
      inserters.push_back(inserter);
    }

    OpInputList components;
    OP_REQUIRES_OK(ctx, ctx->input_list("components", &components));
    for (int i = 0; i < components.size(); ++i) {
      OP_REQUIRES(
          ctx,
          components[i].dims() > 0 && components[i].dim_size(0) == batch_size,
          errors::InvalidArgument(
              "Expected component ", i, " to have batched shape [", batch_size,
              ",...], but saw shape: ", components[i].shape().DebugString()));
    }
    std::vector<TensorShape> element_shapes;
    for (int i = 0; i < components.size(); ++i) {
      TensorShape element_shape = components[i].shape();
      element_shape.RemoveDim(0);
      element_shapes.push_back(std::move(element_shape));
    }
    for (int64 b = 0; b < batch_size; ++b) {
      std::vector<Tensor> tensors;
      tensors.reserve(components.size());
      for (int i = 0; i < components.size(); ++i) {
        Tensor t(components[i].dtype(), element_shapes[i]);
        OP_REQUIRES_OK(ctx,
                       batch_util::CopySliceToElement(components[i], &t, b));
        tensors.push_back(std::move(t));
      }
      // TODO(ebrevdo): Acquire the lock once for all inserters with
      // the same underlying queue?  Add InsertLocked?
      OP_REQUIRES_OK(ctx, inserters[b]->Insert(tensors));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("EnqueueInQueueDataset").Device(DEVICE_CPU),
                        EnqueueInQueueDatasetOp);

}  // namespace

}  // namespace tensorflow
