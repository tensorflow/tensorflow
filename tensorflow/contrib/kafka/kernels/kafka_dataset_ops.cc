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

#include "src-cpp/rdkafkacpp.h"

namespace tensorflow {

class KafkaDatasetOp : public DatasetOpKernel {
 public:
  using DatasetOpKernel::DatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    *output = new Dataset(ctx);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx)
        : GraphDatasetBase(ctx) {}

    std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator({this, strings::StrCat(prefix, "::Kafka")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "KafkaDatasetOp::Dataset"; }

   protected:
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b, Node** output) const override {
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params) : DatasetIterator<Dataset>(params) {}

      Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors, bool* end_of_sequence) override {
        Tensor line_tensor(cpu_allocator(), DT_STRING, {});
        line_tensor.scalar<string>()() = "Hello, world!";

	Tensor arr_tensor(cpu_allocator(), DT_INT32, TensorShape({4}));
	int arr [4] = {1, 2, 3, 4};
	arr_tensor.vec<int32>()(0) = 22;
	arr_tensor.vec<int32>()(1) = 23;
	arr_tensor.vec<int32>()(2) = 24;
	arr_tensor.vec<int32>()(3) = 25;

	Tensor num_tensor(cpu_allocator(), DT_INT64, {});
	num_tensor.scalar<int64>()() = 42;

        out_tensors->emplace_back(std::move(line_tensor));
        out_tensors->emplace_back(std::move(arr_tensor));
        out_tensors->emplace_back(std::move(num_tensor));

        *end_of_sequence = false;
        return Status::OK();
      }

     protected:
      Status SaveInternal(IteratorStateWriter* writer) override {
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx, IteratorStateReader* reader) override {
        return Status::OK();
      }
    };
  };
};

REGISTER_KERNEL_BUILDER(Name("KafkaDataset").Device(DEVICE_CPU), KafkaDatasetOp);

}  // namespace tensorflow
