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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.

// This is based on zip_dataset_op.cc
class UnorderedMergeDatasetOp : public DatasetOpKernel {
 public:
  explicit UnorderedMergeDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    std::vector<DatasetBase*> inputs;
    DatasetBase* input0;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &input0));
    inputs.push_back(input0);
    for (size_t i = 1; i < ctx->num_inputs(); ++i) {
      DatasetBase* input;
      OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
      // from VerifyTypesMatch()
      OP_REQUIRES(
          ctx, input0->output_dtypes().size() == input->output_dtypes().size(),
          errors::InvalidArgument(
              "Number of components does not match: expected ",
              input0->output_dtypes().size(), " types but got ",
              input->output_dtypes().size(), "."));
      for (size_t i = 0; i < input0->output_dtypes().size(); ++i) {
        OP_REQUIRES(ctx,
                    input0->output_dtypes()[i] == input->output_dtypes()[i],
                    errors::InvalidArgument(
                        "Data type mismatch at component ", i, ": expected ",
                        DataTypeString(input0->output_dtypes()[i]), " but got ",
                        DataTypeString(input->output_dtypes()[i]), "."));
      }
      // from VerifyShapesCompatible()
      OP_REQUIRES(
          ctx, input0->output_shapes().size() == input->output_shapes().size(),
          errors::InvalidArgument(
              "Number of components does not match: expected ",
              input0->output_shapes().size(), " types but got ",
              input->output_shapes().size(), "."));
      for (size_t i = 0; i < input0->output_shapes().size(); ++i) {
        OP_REQUIRES(ctx, input0->output_shapes()[i].IsCompatibleWith(
                             input->output_shapes()[i]),
                    errors::InvalidArgument(
                        "Incompatible shapes at component ", i, ": expected ",
                        input0->output_shapes()[i].DebugString(), " but got ",
                        input->output_shapes()[i].DebugString(), "."));
      }
      inputs.push_back(input);
    }
    *output = new Dataset(ctx, inputs);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx,
                     const std::vector<DatasetBase*>& inputs)
        : GraphDatasetBase(ctx), inputs_(inputs) {
      const auto& input = inputs_[0];
      for (DataType dt : input->output_dtypes()) {
        output_dtypes_.push_back(dt);
      }
      output_shapes_.insert(output_shapes_.end(),
                            input->output_shapes().begin(),
                            input->output_shapes().end());
      for (const auto& input : inputs_) {
        input->Ref();
      }
    }

    ~Dataset() override {
      for (const auto& input : inputs_) {
        input->Unref();
      }
    }

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::UnorderedMerge")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_dtypes_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "UnorderedMergeDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(OpKernelContext* ctx, DatasetGraphDefBuilder* b,
                              Node** output) const override {
      std::vector<Node*> input_graph_nodes;
      input_graph_nodes.reserve(inputs_.size());
      for (const auto& input : inputs_) {
        Node* input_node;
        TF_RETURN_IF_ERROR(b->AddParentDataset(ctx, input, &input_node));
        input_graph_nodes.emplace_back(input_node);
      }
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {}, {std::make_pair(0, input_graph_nodes)}, {}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
        input_impls_.reserve(params.dataset->inputs_.size());
        size_t idx = 0;
        for (const auto& input : params.dataset->inputs_) {
          input_impls_.emplace_back(input->MakeIterator(
              strings::StrCat(params.prefix, "[", idx++, "]")));
        }
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (input_impls_.empty()) {
          *end_of_sequence = true;
          return Status::OK();
        }
        out_tensors->clear();
        out_tensors->reserve(1);
        int orig_input_start = input_start;
        do {
          const auto& input_impl = input_impls_[input_start];
          std::vector<Tensor> input_tensors;
          TF_RETURN_IF_ERROR(
              input_impl->GetNext(ctx, &input_tensors, end_of_sequence));
          if (*end_of_sequence == false) {
            out_tensors->insert(out_tensors->end(), input_tensors.begin(),
                                input_tensors.end());
          }
          input_start = (input_start + 1) % input_impls_.size();
        } while (*end_of_sequence && orig_input_start != input_start);
        if (orig_input_start == input_start) {
          *end_of_sequence = true;
        }
        if (*end_of_sequence) {
          out_tensors->clear();
          input_impls_.clear();
        }
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        if (input_impls_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impls_empty"), ""));
        } else {
          for (auto& input_impl : input_impls_)
            TF_RETURN_IF_ERROR(SaveParent(writer, input_impl));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        if (reader->Contains(full_name("input_impls_empty"))) {
          input_impls_.clear();
        } else {
          DCHECK_EQ(input_impls_.size(), dataset()->inputs_.size());
          for (auto& input_impl : input_impls_)
            TF_RETURN_IF_ERROR(RestoreParent(ctx, reader, input_impl));
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::vector<std::unique_ptr<IteratorBase>> input_impls_ GUARDED_BY(mu_);
      int input_start = 0 GUARDED_BY(mu_);  // for round-robin
    };

    const std::vector<DatasetBase*> inputs_;
    DataTypeVector output_dtypes_;
    std::vector<PartialTensorShape> output_shapes_;
  };
};

REGISTER_KERNEL_BUILDER(Name("UnorderedMergeDataset").Device(DEVICE_CPU),
                        UnorderedMergeDatasetOp);

}  // namespace

}  // namespace tensorflow
