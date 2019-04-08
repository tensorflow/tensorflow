/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {

Status DatasetOpsTestBase::ExpectEqual(const Tensor& a, const Tensor& b) {
  EXPECT_EQ(a.dtype(), b.dtype());
  switch (a.dtype()) {
#define CASE(type)                       \
  case DataTypeToEnum<type>::value:      \
    test::ExpectTensorEqual<type>(a, b); \
    break;
    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_string(CASE);
    // TODO(feihugis): figure out how to support variant tensors.
#undef CASE
    default:
      return errors::Internal("Unsupported dtype: ", a.dtype());
  }
  return Status::OK();
}

template <typename T>
bool compare(Tensor t1, Tensor t2) {
  auto flat_t1 = t1.flat<T>();
  auto flat_t2 = t2.flat<T>();
  auto length = std::min(flat_t1.size(), flat_t2.size());
  for (int i = 0; i < length; ++i) {
    if (flat_t1(i) < flat_t2(i)) return true;
    if (flat_t1(i) > flat_t2(i)) return false;
  }
  return flat_t1.size() < length;
}

Status DatasetOpsTestBase::ExpectEqual(std::vector<Tensor> produced_tensors,
                                       std::vector<Tensor> expected_tensors,
                                       bool expect_items_equal) {
  if (produced_tensors.size() != expected_tensors.size()) {
    return Status(tensorflow::errors::Internal(
        "The two tensor vectors have different size (", produced_tensors.size(),
        " v.s. ", expected_tensors.size(), ")"));
  }

  if (produced_tensors.empty()) return Status::OK();
  if (produced_tensors[0].dtype() != expected_tensors[0].dtype()) {
    return Status(tensorflow::errors::Internal(
        "The two tensor vectors have different dtypes (",
        produced_tensors[0].dtype(), " v.s. ", expected_tensors[0].dtype(),
        ")"));
  }

  if (expect_items_equal) {
    const DataType& dtype = produced_tensors[0].dtype();
    switch (dtype) {
#define CASE(DT)                                                \
  case DT:                                                      \
    std::sort(produced_tensors.begin(), produced_tensors.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    std::sort(expected_tensors.begin(), expected_tensors.end(), \
              compare<EnumToDataType<DT>::Type>);               \
    break;
      CASE(DT_FLOAT);
      CASE(DT_DOUBLE);
      CASE(DT_INT32);
      CASE(DT_UINT8);
      CASE(DT_INT16);
      CASE(DT_INT8);
      CASE(DT_STRING);
      CASE(DT_INT64);
      CASE(DT_BOOL);
      CASE(DT_QINT8);
      CASE(DT_QUINT8);
      CASE(DT_QINT32);
      CASE(DT_QINT16);
      CASE(DT_QUINT16);
      CASE(DT_UINT16);
      CASE(DT_HALF);
      CASE(DT_UINT32);
      CASE(DT_UINT64);
      // TODO(feihugis): support other dtypes.
#undef CASE
      default:
        return errors::Internal("Unsupported dtype: ", dtype);
    }
  }

  for (int i = 0; i < produced_tensors.size(); ++i) {
    TF_RETURN_IF_ERROR(DatasetOpsTestBase::ExpectEqual(produced_tensors[i],
                                                       expected_tensors[i]));
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateTensorSliceDatasetKernel(
    StringPiece node_name, const DataTypeVector& dtypes,
    const std::vector<PartialTensorShape>& shapes,
    std::unique_ptr<OpKernel>* tensor_slice_dataset_kernel) {
  std::vector<string> components;
  components.reserve(dtypes.size());
  for (int i = 0; i < dtypes.size(); ++i) {
    // Create the placeholder names for the input components of
    // `TensorSliceDataset`.
    components.emplace_back(strings::StrCat("component_", i));
  }
  NodeDef node_def = test::function::NDef(
      node_name, "TensorSliceDataset", components,
      {{"Toutput_types", dtypes}, {"output_shapes", shapes}});
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, tensor_slice_dataset_kernel));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateTensorSliceDataset(
    StringPiece node_name, std::vector<Tensor>* const components,
    DatasetBase** tensor_slice_dataset) {
  std::unique_ptr<OpKernel> tensor_slice_dataset_kernel;
  DataTypeVector dtypes;
  dtypes.reserve(components->size());
  std::vector<PartialTensorShape> shapes;
  shapes.reserve(components->size());
  for (const auto& t : *components) {
    dtypes.push_back(t.dtype());
    gtl::InlinedVector<int64, 4> partial_dim_sizes;
    for (int i = 1; i < t.dims(); ++i) {
      partial_dim_sizes.push_back(t.dim_size(i));
    }
    shapes.emplace_back(std::move(partial_dim_sizes));
  }
  TF_RETURN_IF_ERROR(CreateTensorSliceDatasetKernel(
      node_name, dtypes, shapes, &tensor_slice_dataset_kernel));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto& tensor : *components) {
    inputs.emplace_back(&tensor);
  }
  TF_RETURN_IF_ERROR(CheckOpKernelInput(*tensor_slice_dataset_kernel, inputs));
  std::unique_ptr<OpKernelContext> context;
  TF_RETURN_IF_ERROR(CreateOpKernelContext(tensor_slice_dataset_kernel.get(),
                                           &inputs, &context));
  TF_RETURN_IF_ERROR(
      RunOpKernel(tensor_slice_dataset_kernel.get(), context.get()));
  TF_RETURN_IF_ERROR(
      GetDatasetFromContext(context.get(), 0, tensor_slice_dataset));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernel(
    const NodeDef& node_def, std::unique_ptr<OpKernel>* op_kernel) {
  OpKernel* kernel;
  TF_RETURN_IF_ERROR(tensorflow::CreateOpKernel(device_type_, device_.get(),
                                                allocator_, flr_, node_def,
                                                TF_GRAPH_DEF_VERSION, &kernel));
  op_kernel->reset(kernel);
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDataset(OpKernel* kernel,
                                         OpKernelContext* context,
                                         DatasetBase** const dataset) {
  TF_RETURN_IF_ERROR(RunOpKernel(kernel, context));
  // Assume that DatasetOp has only one output.
  DCHECK_EQ(context->num_outputs(), 1);
  TF_RETURN_IF_ERROR(GetDatasetFromContext(context, 0, dataset));
  return Status::OK();
}

Status DatasetOpsTestBase::CreateIteratorContext(
    OpKernelContext* const op_context,
    std::unique_ptr<IteratorContext>* iterator_context) {
  IteratorContext::Params params(op_context);
  function_handle_cache_ = absl::make_unique<FunctionHandleCache>(flr_);
  params.function_handle_cache = function_handle_cache_.get();
  *iterator_context = absl::make_unique<IteratorContext>(params);
  return Status::OK();
}

Status DatasetOpsTestBase::GetDatasetFromContext(OpKernelContext* context,
                                                 int output_index,
                                                 DatasetBase** const dataset) {
  Tensor* output = context->mutable_output(output_index);
  Status status = GetDatasetFromVariantTensor(*output, dataset);
  (*dataset)->Ref();
  return status;
}

Status DatasetOpsTestBase::InitThreadPool(int thread_num) {
  if (thread_num < 1) {
    return errors::InvalidArgument(
        "The `thread_num` argument should be positive but got: ", thread_num);
  }
  thread_pool_ = absl::make_unique<thread::ThreadPool>(
      Env::Default(), ThreadOptions(), "inter_op", thread_num);
  return Status::OK();
}

Status DatasetOpsTestBase::InitFunctionLibraryRuntime(
    const std::vector<FunctionDef>& flib, int cpu_num) {
  if (cpu_num < 1) {
    return errors::InvalidArgument(
        "The `cpu_num` argument should be positive but got: ", cpu_num);
  }
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", cpu_num});
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  device_mgr_ = absl::make_unique<DeviceMgr>(std::move(devices));

  FunctionDefLibrary proto;
  for (const auto& fdef : flib) *(proto.add_function()) = fdef;
  lib_def_ =
      absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  OptimizerOptions opts;
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
      opts, thread_pool_.get(), nullptr /* cluster_flr */);
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  if (thread_pool_ == nullptr) {
    runner_ = [](std::function<void()> fn) { fn(); };
  } else {
    runner_ = [this](std::function<void()> fn) {
      thread_pool_->Schedule(std::move(fn));
    };
  }
  return Status::OK();
}

Status DatasetOpsTestBase::RunOpKernel(OpKernel* op_kernel,
                                       OpKernelContext* context) {
  device_->Compute(op_kernel, context);
  return context->status();
}

Status DatasetOpsTestBase::CreateOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
    std::unique_ptr<OpKernelContext>* context) {
  params_ = absl::make_unique<OpKernelContext::Params>();
  params_->device = device_.get();
  params_->resource_manager = device_->resource_manager();
  params_->frame_iter = FrameAndIter(0, 0);
  params_->inputs = inputs;
  params_->op_kernel = kernel;
  params_->function_library = flr_;
  params_->runner = &runner_;
  step_container_ =
      absl::make_unique<ScopedStepContainer>(0, [](const string&) {});
  params_->step_container = step_container_.get();
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
  slice_reader_cache_ =
      absl::make_unique<checkpoint::TensorSliceReaderCacheWrapper>();
  params_->slice_reader_cache = slice_reader_cache_.get();

  // Set the allocator attributes for the outputs.
  allocator_attrs_.clear();
  for (int index = 0; index < params_->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params_->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    allocator_attrs_.emplace_back(attr);
  }
  params_->output_attr_array = gtl::vector_as_array(&allocator_attrs_);

  *context = absl::make_unique<OpKernelContext>(params_.get());
  return Status::OK();
}

Status DatasetOpsTestBase::CreateSerializationContext(
    std::unique_ptr<SerializationContext>* context) {
  *context =
      absl::make_unique<SerializationContext>(SerializationContext::Params{});
  return Status::OK();
}

Status DatasetOpsTestBase::CheckOpKernelInput(
    const OpKernel& kernel, const gtl::InlinedVector<TensorValue, 4>& inputs) {
  if (kernel.input_types().size() != inputs.size()) {
    return errors::Internal("The number of input elements should be ",
                            kernel.input_types().size(),
                            ", but got: ", inputs.size());
  }
  return Status::OK();
}

Status DatasetOpsTestBase::AddDatasetInput(
    gtl::InlinedVector<TensorValue, 4>* inputs, DataTypeVector input_types,
    DataType dtype, const TensorShape& shape) {
  if (input_types.size() < inputs->size()) {
    return errors::InvalidArgument("Adding more inputs than types: ",
                                   inputs->size(), " vs. ", input_types.size());
  }
  bool is_ref = IsRefType(input_types[inputs->size()]);
  std::unique_ptr<Tensor> input =
      absl::make_unique<Tensor>(allocator_, dtype, shape);

  if (is_ref) {
    DataType expected_dtype = RemoveRefType(input_types[inputs->size()]);
    if (expected_dtype != dtype) {
      return errors::InvalidArgument("The input data type is ", dtype,
                                     " , but expected: ", expected_dtype);
    }
    inputs->push_back({&lock_for_refs_, input.get()});
  } else {
    if (input_types[inputs->size()] != dtype) {
      return errors::InvalidArgument(
          "The input data type is ", dtype,
          " , but expected: ", input_types[inputs->size()]);
    }
    inputs->push_back({nullptr, input.get()});
  }

  // TODO(jsimsa): Figure out how to avoid using a member variable to garbage
  // collect the inputs.
  tensors_.push_back(std::move(input));

  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
