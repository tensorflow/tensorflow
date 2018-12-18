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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

class RangeDatasetOpTest : public OpsTestBase {
 public:
  Status InitOp() { return InitOpWithGraphVersion(TF_GRAPH_DEF_VERSION); }

  // Only use this directly if you have a deprecated op that you need to test.
  Status InitOpWithGraphVersion(int graph_def_version) {
    OpKernel* kernel = nullptr;
    Status status = CreateOpKernel(device_type_, device_.get(), allocator(),
                                   flr_, node_def_, graph_def_version, &kernel);
    kernel_.reset(kernel);
    if (kernel_ != nullptr) input_types_ = kernel_->input_types();
    return status;
  }

  Status InitThreadPool(int thread_num = 1) {
    CHECK_GE(thread_num, 1);
    thread_pool_ = new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                          "inter_op", thread_num);
    return Status::OK();
  }

  Status InitFunctionLibraryRuntime(const std::vector<FunctionDef>& flib,
                                    thread::ThreadPool* thread_pool = nullptr,
                                    int cpu_num = 2) {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", cpu_num});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));
    device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));

    FunctionDefLibrary proto;
    for (const auto& fdef : flib) *(proto.add_function()) = fdef;
    lib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), proto));
    fdef_lib_ = lib_def_->ToProto();

    OptimizerOptions opts;
    pflr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
        opts, thread_pool, nullptr /* cluster_flr */));

    flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");

    if (thread_pool == nullptr) {
      runner_ = [](std::function<void()> fn) { fn(); };
    } else {
      runner_ = [thread_pool](std::function<void()> fn) {
        thread_pool->Schedule(std::move(fn));
      };
    }

    return Status::OK();
  }

  Status GetDatasetOutputFromContext(int output_index) {
    auto* tensor = GetOutput(output_index);
    TF_CHECK_OK(GetDatasetFromVariantTensor(*tensor, &dataset_));
    return Status::OK();
  }

  Status RunOpKernel() {
    // Make sure the old OpKernelContext is deleted before the Params it was
    // using.
    context_.reset(nullptr);
    params_.reset(new OpKernelContext::Params);
    params_.get()->device = device_.get();
    params_.get()->frame_iter = FrameAndIter(0, 0);
    params_.get()->inputs = &inputs_;
    params_.get()->op_kernel = kernel_.get();
    params_.get()->function_library = flr_;
    params_.get()->runner = &runner_;

    step_container_.reset(new ScopedStepContainer(0, [](const string&) {}));
    params_->step_container = step_container_.get();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(params_.get(), &attrs);
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params_.get()->slice_reader_cache = &slice_reader_cache_wrapper;
    params_.get()->resource_manager = device_.get()->resource_manager();

    context_.reset(new OpKernelContext(params_.get()));
    device_->Compute(kernel_.get(), context_.get());
    return context_->status();
  }

 protected:
  template <typename T>
  void MakeOpDef() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    DataTypeVector* dtypes = new DataTypeVector({value_type});

    TF_CHECK_OK(NodeDefBuilder("rangedataset", "RangeDataset")
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Input(FakeInput(DT_INT64))
                    .Attr("output_types", *dtypes)
                    .Attr("output_shapes", *shapes)
                    .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  Status MakeDataset(int64 start, int64 end, int64 step, int output_index = 0,
                     int thread_num = 2) {
    AddInputFromArray<int64>(TensorShape({}), {start});
    AddInputFromArray<int64>(TensorShape({}), {end});
    AddInputFromArray<int64>(TensorShape({}), {step});

    TF_CHECK_OK(InitThreadPool(thread_num));
    TF_CHECK_OK(InitFunctionLibraryRuntime({}, thread_pool_));
    TF_CHECK_OK(RunOpKernel());

    TF_CHECK_OK(GetDatasetOutputFromContext(output_index));
    return Status::OK();
  }

  Status MakeIteratorContext() {
    iteratorContext_ = new IteratorContext(context_.get());
    return Status::OK();
  }

  Status MakeIterator() {
    iterator_.reset(nullptr);
    TF_CHECK_OK(
        dataset_->MakeIterator(iteratorContext_, "Iterator", &iterator_));
    return Status::OK();
  }

  Status GetNext(std::vector<Tensor>* out_tensors, bool* end_of_sequence) {
    TF_CHECK_OK(
        iterator_->GetNext(iteratorContext_, out_tensors, end_of_sequence));
    return Status::OK();
  }

 protected:
  std::unique_ptr<DeviceMgr> device_mgr_;
  FunctionLibraryRuntime* flr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionDefLibrary fdef_lib_;
  thread::ThreadPool* thread_pool_;
  std::function<void(std::function<void()>)> runner_;
  DatasetBase* dataset_;
  IteratorContext* iteratorContext_;
  std::unique_ptr<IteratorBase> iterator_;
};

TEST_F(RangeDatasetOpTest, GetNext) {
  MakeOpDef<int64>();
  int start = 0, end = 10, step = 1;
  TF_CHECK_OK(MakeDataset(start, end, step));
  TF_CHECK_OK(MakeIteratorContext());
  TF_CHECK_OK(MakeIterator());
  bool end_of_sequence = false;
  std::vector<Tensor>* out_tensors = new std::vector<Tensor>();

  while (!end_of_sequence) {
    TF_CHECK_OK(GetNext(out_tensors, &end_of_sequence));
  }

  std::vector<int64> expected_values;
  for (int i = start; i < end; i = i + step) {
    expected_values.emplace_back(i);
  }

  EXPECT_EQ(out_tensors->size(), out_tensors->size());

  for (size_t i = 0; i < out_tensors->size(); ++i) {
    int64 actual_value = out_tensors->at(i).flat<int64>()(0);
    int64 expect_value = expected_values[i];
    EXPECT_EQ(actual_value, expect_value);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
