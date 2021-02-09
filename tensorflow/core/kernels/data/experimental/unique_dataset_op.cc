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
#include "tensorflow/core/kernels/data/experimental/unique_dataset_op.h"

#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const UniqueDatasetOp::kDatasetType;
/* static */ constexpr const char* const UniqueDatasetOp::kInputDataset;
/* static */ constexpr const char* const UniqueDatasetOp::kOutputTypes;
/* static */ constexpr const char* const UniqueDatasetOp::kOutputShapes;

class UniqueDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)), input_(input) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::Unique")});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return strings::StrCat("UniqueDatasetOp::Dataset");
  }

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
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const typename Iterator::Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      mutex_lock l(mu_);
      bool saw_new_value;
      do {
        saw_new_value = false;
        out_tensors->clear();
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (*end_of_sequence) {
          break;
        }
        DCHECK_EQ(1, out_tensors->size());
        saw_new_value = unique_elements_.insert((*out_tensors)[0]).second;
      } while (!saw_new_value);
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("input_impl_empty"), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("unique_elements_size"),
                                             unique_elements_.size()));
      size_t i = 0;
      for (const Tensor& t : unique_elements_) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            full_name(strings::StrCat("unique_elements[", i++, "]")), t));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      if (!reader->Contains(full_name("input_impl_empty"))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      int64 num_unique_elements;
      unique_elements_.clear();
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("unique_elements_size"),
                                            &num_unique_elements));
      for (int64 i = 0; i < num_unique_elements; ++i) {
        Tensor unique_element;
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            full_name(strings::StrCat("unique_elements[", i, "]")),
            &unique_element));
        auto insert_result = unique_elements_.insert(unique_element);
        if (!insert_result.second) {
          return errors::InvalidArgument(
              "Checkpoint contained two unique elements with the same "
              "value.");
        }
      }
      return Status::OK();
    }

   private:
    struct TensorHash {
      size_t operator()(const Tensor& t) const {
        if (t.dtype() == DT_INT32 || t.dtype() == DT_INT64) {
          return Hash64(t.tensor_data().data(), t.tensor_data().size());
        } else {
          DCHECK_EQ(DT_STRING, t.dtype());
          auto flat_t = t.flat<tstring>();
          uint64 hash = 0;
          for (int64 i = 0; i < t.NumElements(); ++i) {
            hash = Hash64Combine(hash, Hash64(flat_t(i)));
          }
          return static_cast<size_t>(hash);
        }
      }
    };

    struct TensorKeyEqual {
      bool operator()(const Tensor& lhs, const Tensor& rhs) const {
        if (lhs.shape() != rhs.shape() || lhs.dtype() != rhs.dtype()) {
          return false;
        }
        switch (lhs.dtype()) {
#define HANDLE_TYPE(T)                                     \
  case T:                                                  \
    do {                                                   \
      auto lhs_flat = lhs.flat<EnumToDataType<T>::Type>(); \
      auto rhs_flat = rhs.flat<EnumToDataType<T>::Type>(); \
      for (int64 i = 0; i < lhs.NumElements(); ++i) {      \
        if (lhs_flat(i) != rhs_flat(i)) {                  \
          return false;                                    \
        }                                                  \
      }                                                    \
      return true;                                         \
    } while (0)

            HANDLE_TYPE(DT_INT32);
            HANDLE_TYPE(DT_INT64);
            HANDLE_TYPE(DT_STRING);
            default:
              DCHECK(false) << "UniqueDataset unhandled data type: "
                            << DataTypeString(lhs.dtype());
              return false;
        }
      }
    };

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      std::unordered_set<Tensor, TensorHash, TensorKeyEqual> unique_elements_
          TF_GUARDED_BY(mu_);
  };

    const DatasetBase* const input_;
};

void UniqueDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
  OP_REQUIRES(ctx, input->output_dtypes().size() == 1,
              errors::InvalidArgument("UniqueDataset only supports "
                                      "inputs with a single component."));

  DataType input_dtype = input->output_dtypes()[0];
  OP_REQUIRES(ctx,
              input_dtype == DT_INT32 || input_dtype == DT_INT64 ||
                  input_dtype == DT_STRING,
              errors::InvalidArgument(
                  "UniqueDataset only supports inputs with a single "
                  "`tf.int32`, `tf.int64`, or `tf.string` component."));

  *output = new Dataset(ctx, input);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("UniqueDataset").Device(DEVICE_CPU),
                        UniqueDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalUniqueDataset").Device(DEVICE_CPU),
                        UniqueDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
