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
#include <map>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/dataset.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class GroupByReducerDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit GroupByReducerDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key_func", &key_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("init_func", &init_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_func", &reduce_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("finalize_func", &finalize_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
    std::unique_ptr<CapturedFunction> captured_key_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(key_func_, ctx,
                                                 "key_func_other_arguments",
                                                 &captured_key_func));
    std::unique_ptr<CapturedFunction> captured_init_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(init_func_, ctx,
                                                 "init_func_other_arguments",
                                                 &captured_init_func));
    std::unique_ptr<CapturedFunction> captured_reduce_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(reduce_func_, ctx,
                                                 "reduce_func_other_arguments",
                                                 &captured_reduce_func));
    std::unique_ptr<CapturedFunction> captured_finalize_func;
    OP_REQUIRES_OK(ctx,
                   CapturedFunction::Create(finalize_func_, ctx,
                                            "finalize_func_other_arguments",
                                            &captured_finalize_func));

    *output = new Dataset(
        ctx, input, std::move(captured_key_func), std::move(captured_init_func),
        std::move(captured_reduce_func), std::move(captured_finalize_func),
        output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_key_func,
            std::unique_ptr<CapturedFunction> captured_init_func,
            std::unique_ptr<CapturedFunction> captured_reduce_func,
            std::unique_ptr<CapturedFunction> captured_finalize_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          captured_key_func_(std::move(captured_key_func)),
          captured_init_func_(std::move(captured_init_func)),
          captured_reduce_func_(std::move(captured_reduce_func)),
          captured_finalize_func_(std::move(captured_finalize_func)),
          output_types_(output_types),
          output_shapes_(output_shapes) {
      input_->Ref();
    }

    ~Dataset() override { input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::GroupByReducer")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() const override {
      return "GroupByReducerDatasetOp::Dataset";
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, key_func().name()));
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, init_func().name()));
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, reduce_func().name()));
      TF_RETURN_IF_ERROR(b->AddFunction(ctx, finalize_func().name()));
      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      std::vector<Node*> key_func_other_arguments_node;
      DataTypeVector key_func_other_arguments_types;
      TF_RETURN_IF_ERROR(OtherArgumentsNodeAndType(
          b, captured_key_func_, &key_func_other_arguments_node,
          &key_func_other_arguments_types));

      std::vector<Node*> init_func_other_arguments_node;
      DataTypeVector init_func_other_arguments_types;
      TF_RETURN_IF_ERROR(OtherArgumentsNodeAndType(
          b, captured_init_func_, &init_func_other_arguments_node,
          &init_func_other_arguments_types));

      std::vector<Node*> reduce_func_other_arguments_node;
      DataTypeVector reduce_func_other_arguments_types;
      TF_RETURN_IF_ERROR(OtherArgumentsNodeAndType(
          b, captured_reduce_func_, &reduce_func_other_arguments_node,
          &reduce_func_other_arguments_types));

      std::vector<Node*> finalize_func_other_arguments_node;
      DataTypeVector finalize_func_other_arguments_types;
      TF_RETURN_IF_ERROR(OtherArgumentsNodeAndType(
          b, captured_finalize_func_, &finalize_func_other_arguments_node,
          &finalize_func_other_arguments_types));

      AttrValue key_func;
      b->BuildAttrValue(this->key_func(), &key_func);
      AttrValue init_func;
      b->BuildAttrValue(this->init_func(), &init_func);
      AttrValue reduce_func;
      b->BuildAttrValue(this->reduce_func(), &reduce_func);
      AttrValue finalize_func;
      b->BuildAttrValue(this->finalize_func(), &finalize_func);

      AttrValue key_func_other_arguments_types_attr;
      b->BuildAttrValue(key_func_other_arguments_types,
                        &key_func_other_arguments_types_attr);
      AttrValue init_func_other_arguments_types_attr;
      b->BuildAttrValue(init_func_other_arguments_types,
                        &init_func_other_arguments_types_attr);
      AttrValue reduce_func_other_arguments_types_attr;
      b->BuildAttrValue(reduce_func_other_arguments_types,
                        &reduce_func_other_arguments_types_attr);
      AttrValue finalize_func_other_arguments_types_attr;
      b->BuildAttrValue(finalize_func_other_arguments_types,
                        &finalize_func_other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {{0, input_graph_node}},
          {{1, key_func_other_arguments_node},
           {2, init_func_other_arguments_node},
           {3, reduce_func_other_arguments_node},
           {4, finalize_func_other_arguments_node}},
          {{"key_func", key_func},
           {"init_func", init_func},
           {"reduce_func", reduce_func},
           {"finalize_func", finalize_func},
           {"Tkey_func_other_arguments", key_func_other_arguments_types_attr},
           {"Tinit_func_other_arguments", init_func_other_arguments_types_attr},
           {"Treduce_func_other_arguments",
            reduce_func_other_arguments_types_attr},
           {"Tfinalize_func_other_arguments",
            finalize_func_other_arguments_types_attr}},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {}

      Status Initialize(IteratorContext* ctx) override {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(dataset()->captured_key_func_->Instantiate(ctx));
        TF_RETURN_IF_ERROR(dataset()->captured_init_func_->Instantiate(ctx));
        TF_RETURN_IF_ERROR(dataset()->captured_reduce_func_->Instantiate(ctx));
        TF_RETURN_IF_ERROR(
            dataset()->captured_finalize_func_->Instantiate(ctx));
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);

        // Iterate through the input dataset, keying input elements to reducers.
        while (!end_of_input_) {
          std::vector<Tensor> next_input_element;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &next_input_element, &end_of_input_));

          if (!end_of_input_) {
            // Run the key function on the input element.
            std::vector<Tensor> key_func_output;
            TF_RETURN_IF_ERROR(
                dataset()->captured_key_func_->RunWithBorrowedArgs(
                    ctx, next_input_element, &key_func_output));

            if (key_func_output.size() != 1 ||
                key_func_output[0].dtype() != DT_INT64 ||
                key_func_output[0].NumElements() != 1) {
              // TODO(b/78665031): Support non-int64 keys.
              return errors::InvalidArgument(
                  "`key_func` must return a scalar int64.");
            }
            const int64 key = key_func_output[0].scalar<int64>()();

            if (states_.find(key) == states_.end()) {
              // Run the init function to create the initial state.
              std::vector<Tensor> init_func_output;
              TF_RETURN_IF_ERROR(dataset()->captured_init_func_->Run(
                  ctx, std::move(key_func_output), &init_func_output));
              states_[key] = init_func_output;
            }

            // Run the reduce function to update the current state.
            std::vector<Tensor> args;
            args.reserve(states_[key].size() + next_input_element.size());
            std::copy(states_[key].begin(), states_[key].end(),
                      std::back_inserter(args));
            std::copy(next_input_element.begin(), next_input_element.end(),
                      std::back_inserter(args));

            std::vector<Tensor> reduce_func_output;
            TF_RETURN_IF_ERROR(dataset()->captured_reduce_func_->Run(
                ctx, std::move(args), &reduce_func_output));
            states_[key] = reduce_func_output;
          } else {
            keys_.resize(states_.size());
            int idx = 0;
            for (auto it = states_.begin(); it != states_.end(); ++idx, ++it) {
              keys_[idx] = it->first;
            }
          }
        }

        if (keys_index_ == keys_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }
        TF_RETURN_IF_ERROR(
            dataset()->captured_finalize_func_->RunWithBorrowedArgs(
                ctx, states_[keys_[keys_index_++]], out_tensors));
        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));

        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }

        // Saving states_.
        if (!states_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("states_size"), states_.size()));
          int idx = 0;
          for (auto it = states_.begin(); it != states_.end(); ++idx, ++it) {
            int64 key = it->first;
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("states[", idx, "]->key")), key));
            if (!it->second.empty()) {
              TF_RETURN_IF_ERROR(writer->WriteScalar(
                  full_name(strings::StrCat("states[", idx, "]->state_size")),
                  it->second.size()));
              for (int j = 0; j < it->second.size(); ++j) {
                TF_RETURN_IF_ERROR(writer->WriteTensor(
                    full_name(
                        strings::StrCat("states[", idx, "]->state[", j, "]")),
                    it->second[j]));
              }
            }
          }
        }

        // Saving keys_index_ and keys_.
        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("keys_index"), keys_index_));
          if (!keys_.empty()) {
            TF_RETURN_IF_ERROR(
                writer->WriteScalar(full_name("keys_size"), keys_.size()));
            for (int idx = 0; idx < keys_.size(); ++idx) {
              TF_RETURN_IF_ERROR(writer->WriteScalar(
                  full_name(strings::StrCat("keys[", idx, "]")), keys_[idx]));
            }
          }
        }

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;

        // Restoring states_.
        if (reader->Contains(full_name("states_size"))) {
          int64 size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("states_size"), &size));
          for (int idx = 0; idx < size; ++idx) {
            int64 key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("states[", idx, "]->key")), &key));
            std::vector<Tensor> state;
            if (reader->Contains(full_name(
                    strings::StrCat("states[", idx, "]->state_size")))) {
              int64 state_size;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name(strings::StrCat("states[", idx, "]->state_size")),
                  &state_size));
              state.resize(state_size);
              for (int j = 0; j < state_size; ++j) {
                TF_RETURN_IF_ERROR(reader->ReadTensor(
                    full_name(
                        strings::StrCat("states[", idx, "]->state[", j, "]")),
                    &state[j]));
              }
            }
            states_[key] = state;
          }
        }

        // Restoring keys_index_ and keys_.
        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("keys_index"), &keys_index_));
          if (reader->Contains(full_name("keys_size"))) {
            int64 size;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name("keys_size"), &size));
            keys_.resize(size);
            for (int idx = 0; idx < size; ++idx) {
              int64 key;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name(strings::StrCat("keys[", idx, "]")), &key));
              keys_[idx] = key;
            }
          }
        }

        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
      bool end_of_input_ GUARDED_BY(mu_) = false;
      std::map<int64, std::vector<Tensor>> states_ GUARDED_BY(mu_);
      std::vector<int64> keys_ GUARDED_BY(mu_);
      int64 keys_index_ GUARDED_BY(mu_) = 0;
    };

    const NameAttrList& key_func() const { return captured_key_func_->func(); }

    const NameAttrList& init_func() const {
      return captured_init_func_->func();
    }

    const NameAttrList& reduce_func() const {
      return captured_reduce_func_->func();
    }

    const NameAttrList& finalize_func() const {
      return captured_finalize_func_->func();
    }

    Status OtherArgumentsNodeAndType(
        DatasetGraphDefBuilder* b,
        const std::unique_ptr<CapturedFunction>& captured_func,
        std::vector<Node*>* other_arguments_node,
        DataTypeVector* other_arguments_types) const {
      other_arguments_node->reserve(captured_func->captured_inputs().size());
      other_arguments_types->reserve(captured_func->captured_inputs().size());
      for (const Tensor& t : captured_func->captured_inputs()) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        other_arguments_node->emplace_back(node);
        other_arguments_types->emplace_back(t.dtype());
      }
      return Status::OK();
    }

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_key_func_;
    const std::unique_ptr<CapturedFunction> captured_init_func_;
    const std::unique_ptr<CapturedFunction> captured_reduce_func_;
    const std::unique_ptr<CapturedFunction> captured_finalize_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  NameAttrList key_func_;
  NameAttrList init_func_;
  NameAttrList reduce_func_;
  NameAttrList finalize_func_;
};

REGISTER_KERNEL_BUILDER(Name("GroupByReducerDataset").Device(DEVICE_CPU),
                        GroupByReducerDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
