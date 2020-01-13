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

#include <memory>

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/kernels/data/batch_dataset_op.h"
#include "tensorflow/core/kernels/data/concatenate_dataset_op.h"
#include "tensorflow/core/kernels/data/map_dataset_op.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {

string ToString(CompressionType compression_type) {
  switch (compression_type) {
    case CompressionType::ZLIB:
      return "ZLIB";
    case CompressionType::GZIP:
      return "GZIP";
    case CompressionType::RAW:
      return "RAW";
    case CompressionType::UNCOMPRESSED:
      return "";
  }
}

io::ZlibCompressionOptions GetZlibCompressionOptions(
    CompressionType compression_type) {
  switch (compression_type) {
    case CompressionType::ZLIB:
      return io::ZlibCompressionOptions::DEFAULT();
    case CompressionType::GZIP:
      return io::ZlibCompressionOptions::GZIP();
    case CompressionType::RAW:
      return io::ZlibCompressionOptions::RAW();
    case CompressionType::UNCOMPRESSED:
      LOG(WARNING) << "ZlibCompressionOptions does not have an option for "
                   << ToString(compression_type);
      return io::ZlibCompressionOptions::DEFAULT();
  }
}

Status WriteDataToFile(const string& filename, const char* data) {
  return WriteDataToFile(filename, data, CompressionParams());
}

Status WriteDataToFile(const string& filename, const char* data,
                       const CompressionParams& params) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file_writer;
  TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file_writer));
  if (params.compression_type == CompressionType::UNCOMPRESSED) {
    TF_RETURN_IF_ERROR(file_writer->Append(data));
  } else if (params.compression_type == CompressionType::ZLIB ||
             params.compression_type == CompressionType::GZIP ||
             params.compression_type == CompressionType::RAW) {
    auto zlib_compression_options =
        GetZlibCompressionOptions(params.compression_type);
    io::ZlibOutputBuffer out(file_writer.get(), params.input_buffer_size,
                             params.output_buffer_size,
                             zlib_compression_options);
    TF_RETURN_IF_ERROR(out.Init());
    TF_RETURN_IF_ERROR(out.Append(data));
    TF_RETURN_IF_ERROR(out.Flush());
    TF_RETURN_IF_ERROR(out.Close());
  } else {
    return tensorflow::errors::InvalidArgument(
        "Unsupported compression_type: ", ToString(params.compression_type));
  }

  TF_RETURN_IF_ERROR(file_writer->Flush());
  TF_RETURN_IF_ERROR(file_writer->Close());

  return Status::OK();
}

Status WriteDataToTFRecordFile(const string& filename,
                               const std::vector<absl::string_view>& records,
                               const CompressionParams& params) {
  Env* env = Env::Default();
  std::unique_ptr<WritableFile> file_writer;
  TF_RETURN_IF_ERROR(env->NewWritableFile(filename, &file_writer));
  auto options = io::RecordWriterOptions::CreateRecordWriterOptions(
      ToString(params.compression_type));
  options.zlib_options.input_buffer_size = params.input_buffer_size;
  io::RecordWriter record_writer(file_writer.get(), options);
  for (const auto& record : records) {
    TF_RETURN_IF_ERROR(record_writer.WriteRecord(record));
  }
  TF_RETURN_IF_ERROR(record_writer.Flush());
  TF_RETURN_IF_ERROR(record_writer.Close());
  TF_RETURN_IF_ERROR(file_writer->Flush());
  TF_RETURN_IF_ERROR(file_writer->Close());
  return Status::OK();
}

template <typename T>
Status IsEqual(const Tensor& t1, const Tensor& t2) {
  if (t1.dtype() != t2.dtype()) {
    return tensorflow::errors::Internal(
        "Two tensors have different dtypes: ", DataTypeString(t1.dtype()),
        " vs. ", DataTypeString(t2.dtype()));
  }
  if (!t1.IsSameSize(t2)) {
    return tensorflow::errors::Internal(
        "Two tensors have different shapes: ", t1.shape().DebugString(),
        " vs. ", t2.shape().DebugString());
  }

  auto flat_t1 = t1.flat<T>();
  auto flat_t2 = t2.flat<T>();
  auto length = flat_t1.size();

  for (int i = 0; i < length; ++i) {
    if (flat_t1(i) != flat_t2(i)) {
      return tensorflow::errors::Internal(
          "Two tensors have different values "
          "at [",
          i, "]: ", flat_t1(i), " vs. ", flat_t2(i));
    }
  }
  return Status::OK();
}

Status DatasetOpsTestBase::ExpectEqual(const Tensor& a, const Tensor& b) {
  switch (a.dtype()) {
#define CASE(DT)                           \
  case DataTypeToEnum<DT>::value:          \
    TF_RETURN_IF_ERROR(IsEqual<DT>(a, b)); \
    break;
    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_tstring(CASE);
    TF_CALL_uint32(CASE);
    TF_CALL_uint64(CASE);
    // TODO(feihugis): figure out how to support variant tensors.
#undef CASE
    default:
      return errors::Internal("Unsupported dtype: ", a.dtype());
  }
  return Status::OK();
}

template <typename T>
bool compare(const Tensor& t1, const Tensor& t2) {
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
                                       bool compare_order) {
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

  if (!compare_order) {
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

Status DatasetOpsTestBase::CreateOpKernel(
    const NodeDef& node_def, std::unique_ptr<OpKernel>* op_kernel) {
  OpKernel* kernel;
  TF_RETURN_IF_ERROR(tensorflow::CreateOpKernel(device_type_, device_.get(),
                                                allocator_, flr_, node_def,
                                                TF_GRAPH_DEF_VERSION, &kernel));
  op_kernel->reset(kernel);
  return Status::OK();
}

Status DatasetOpsTestBase::CreateDatasetContext(
    OpKernel* const dateset_kernel,
    gtl::InlinedVector<TensorValue, 4>* const inputs,
    std::unique_ptr<OpKernelContext::Params>* dataset_context_params,
    std::unique_ptr<OpKernelContext>* dataset_context) {
  TF_RETURN_IF_ERROR(CheckOpKernelInput(*dateset_kernel, *inputs));
  TF_RETURN_IF_ERROR(CreateOpKernelContext(
      dateset_kernel, inputs, dataset_context_params, dataset_context));
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

Status DatasetOpsTestBase::RestoreIterator(
    IteratorContext* ctx, IteratorStateReader* reader,
    const string& output_prefix, const DatasetBase& dataset,
    std::unique_ptr<IteratorBase>* iterator) {
  return dataset.MakeIteratorFromCheckpoint(ctx, output_prefix, reader,
                                            iterator);
}

Status DatasetOpsTestBase::CreateIteratorContext(
    OpKernelContext* const op_context,
    std::unique_ptr<IteratorContext>* iterator_context) {
  IteratorContext::Params params(op_context);
  params.resource_mgr = op_context->resource_manager();
  function_handle_cache_ = absl::make_unique<FunctionHandleCache>(flr_);
  params.function_handle_cache = function_handle_cache_.get();
  params.cancellation_manager = cancellation_manager_.get();
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
      Env::Default(), ThreadOptions(), "test_thread_pool", thread_num);
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
  device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
  resource_mgr_ = absl::make_unique<ResourceMgr>("default_container");

  FunctionDefLibrary proto;
  for (const auto& fdef : flib) *(proto.add_function()) = fdef;
  lib_def_ =
      absl::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  OptimizerOptions opts;
  pflr_ = absl::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, lib_def_.get(), opts, thread_pool_.get(),
      nullptr /* cluster_flr */);
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  if (thread_pool_ == nullptr) {
    runner_ = [](const std::function<void()>& fn) { fn(); };
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

Status DatasetOpsTestBase::RunFunction(
    const FunctionDef& fdef, test::function::Attrs attrs,
    const std::vector<Tensor>& args,
    const GraphConstructorOptions& graph_options, std::vector<Tensor*> rets) {
  std::unique_ptr<Executor> exec;
  InstantiationResult result;
  auto GetOpSig = [](const string& op, const OpDef** sig) {
    return OpRegistry::Global()->LookUpOpDef(op, sig);
  };
  TF_RETURN_IF_ERROR(InstantiateFunction(fdef, attrs, GetOpSig, &result));

  DataTypeVector arg_types = result.arg_types;
  DataTypeVector ret_types = result.ret_types;

  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  TF_RETURN_IF_ERROR(
      ConvertNodeDefsToGraph(graph_options, result.nodes, g.get()));

  const int version = g->versions().producer();
  LocalExecutorParams params;
  params.function_library = flr_;
  params.device = device_.get();
  params.create_kernel = [this, version](const NodeDef& ndef,
                                         OpKernel** kernel) {
    return CreateNonCachedKernel(device_.get(), this->flr_, ndef, version,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };
  params.rendezvous_factory = [](const int64, const DeviceMgr* device_mgr,
                                 Rendezvous** r) {
    *r = new IntraProcessRendezvous(device_mgr);
    return Status::OK();
  };

  Executor* cur_exec;
  TF_RETURN_IF_ERROR(NewLocalExecutor(params, *g, &cur_exec));
  exec.reset(cur_exec);
  FunctionCallFrame frame(arg_types, ret_types);
  TF_RETURN_IF_ERROR(frame.SetArgs(args));
  Executor::Args exec_args;
  exec_args.call_frame = &frame;
  exec_args.runner = runner_;
  TF_RETURN_IF_ERROR(exec->Run(exec_args));
  std::vector<Tensor> computed;
  TF_RETURN_IF_ERROR(frame.GetRetvals(&computed));
  if (computed.size() != rets.size()) {
    return errors::InvalidArgument(
        "The result does not match the expected number of return outpus",
        ". Expected: ", rets.size(), ". Actual: ", computed.size());
  }
  for (int i = 0; i < rets.size(); ++i) {
    *(rets[i]) = computed[i];
  }
  return Status::OK();
}

Status DatasetOpsTestBase::CreateOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
    std::unique_ptr<OpKernelContext>* context) {
  return CreateOpKernelContext(kernel, inputs, &params_, context);
}

Status DatasetOpsTestBase::CreateOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
    std::unique_ptr<OpKernelContext::Params>* context_params,
    std::unique_ptr<OpKernelContext>* context) {
  auto params = absl::make_unique<OpKernelContext::Params>();
  cancellation_manager_ = absl::make_unique<CancellationManager>();
  params->cancellation_manager = cancellation_manager_.get();
  params->device = device_.get();
  params->frame_iter = FrameAndIter(0, 0);
  params->function_library = flr_;
  params->inputs = inputs;
  params->op_kernel = kernel;
  params->resource_manager = resource_mgr_.get();
  params->runner = &runner_;
  slice_reader_cache_ =
      absl::make_unique<checkpoint::TensorSliceReaderCacheWrapper>();
  params->slice_reader_cache = slice_reader_cache_.get();
  step_container_ =
      absl::make_unique<ScopedStepContainer>(0, [](const string&) {});
  params->step_container = step_container_.get();

  // Set the allocator attributes for the outputs.
  allocator_attrs_.clear();
  for (int index = 0; index < params->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    allocator_attrs_.emplace_back(attr);
  }
  params->output_attr_array = allocator_attrs_.data();

  *context = absl::make_unique<OpKernelContext>(params.get());
  *context_params = std::move(params);
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

Status DatasetOpsTestBase::CheckIteratorGetNext(
    const std::vector<Tensor>& expected_outputs, bool compare_order) {
  return CheckIteratorGetNext(iterator_.get(), iterator_ctx_.get(),
                              expected_outputs, compare_order);
}

Status DatasetOpsTestBase::CheckIteratorGetNext(
    TestIterator* iterator, const std::vector<Tensor>& expected_outputs,
    bool compare_order) {
  return CheckIteratorGetNext(iterator->iterator(), iterator->ctx(),
                              expected_outputs, compare_order);
}

Status DatasetOpsTestBase::CheckIteratorGetNext(
    IteratorBase* iterator, IteratorContext* ctx,
    const std::vector<Tensor>& expected_outputs, bool compare_order) {
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order=*/compare_order));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetNodeName(
    const string& expected_dataset_node_name) {
  EXPECT_EQ(dataset_->node_name(), expected_dataset_node_name);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetTypeString(
    const string& expected_type_str) {
  EXPECT_EQ(dataset_->type_string(), expected_type_str);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
  TF_EXPECT_OK(
      VerifyTypesMatch(dataset_->output_dtypes(), expected_output_dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(dataset_->output_shapes(),
                                      expected_output_shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckDatasetCardinality(int expected_cardinality) {
  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
  TF_EXPECT_OK(
      VerifyTypesMatch(iterator_->output_dtypes(), expected_output_dtypes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(iterator_->output_shapes(),
                                      expected_output_shapes));
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorPrefix(
    const string& expected_iterator_prefix) {
  EXPECT_EQ(iterator_->prefix(), expected_iterator_prefix);
  return Status::OK();
}

Status DatasetOpsTestBase::CheckIteratorSaveAndRestore(
    const string& iterator_prefix, const std::vector<Tensor>& expected_outputs,
    const std::vector<int>& breakpoints, bool compare_order) {
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(
      dataset_->MakeIterator(iterator_ctx_.get(), iterator_prefix, &iterator));
  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_RETURN_IF_ERROR(CreateSerializationContext(&serialization_ctx));
  bool end_of_sequence = false;
  int cur_iteration = 0;
  std::vector<Tensor> out_tensors;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader, iterator_prefix,
                                 *dataset_, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_RETURN_IF_ERROR(
          iterator->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }

    if (dataset_->Cardinality() == kUnknownCardinality) {
      continue;
    }

    if (dataset_->Cardinality() == kInfiniteCardinality ||
        breakpoint < dataset_->Cardinality()) {
      EXPECT_FALSE(end_of_sequence);
    } else {
      EXPECT_TRUE(end_of_sequence);
    }
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order=*/compare_order));
  return Status::OK();
}

Status DatasetOpsTestBase::Initialize(const DatasetParams& dataset_params) {
  if (initialized_) {
    return errors::Internal(
        "The fields (e.g. dataset_kernel_, dataset_ctx_, dataset_, "
        "iterator_ctx_, iterator_) have already been initialized.");
  }
  TF_RETURN_IF_ERROR(InitializeRuntime(dataset_params));
  TF_RETURN_IF_ERROR(MakeDataset(dataset_params, &dataset_kernel_, &params_,
                                 &dataset_ctx_, &tensors_, &dataset_));
  TF_RETURN_IF_ERROR(CreateIteratorContext(dataset_ctx_.get(), &iterator_ctx_));
  TF_RETURN_IF_ERROR(dataset_->MakeIterator(
      iterator_ctx_.get(), dataset_params.iterator_prefix(), &iterator_));
  initialized_ = true;
  return Status::OK();
}

Status DatasetOpsTestBase::InitializeRuntime(
    const DatasetParams& dataset_params) {
  TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
  TF_RETURN_IF_ERROR(
      InitFunctionLibraryRuntime(dataset_params.func_lib(), cpu_num_));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDataset(const DatasetParams& dataset_params,
                                       std::unique_ptr<TestDataset>* dataset) {
  DatasetBase* dataset_base;
  std::unique_ptr<OpKernel> dataset_kernel;
  std::unique_ptr<OpKernelContext::Params> dataset_ctx_params;
  std::unique_ptr<OpKernelContext> dataset_ctx;
  std::vector<std::unique_ptr<Tensor>> created_tensors;
  TF_RETURN_IF_ERROR(MakeDataset(dataset_params, &dataset_kernel,
                                 &dataset_ctx_params, &dataset_ctx,
                                 &created_tensors, &dataset_base));
  *dataset = std::make_unique<TestDataset>(
      std::move(dataset_kernel), std::move(dataset_ctx_params),
      std::move(dataset_ctx), std::move(created_tensors), dataset_base);
  return Status::OK();
}

Status DatasetOpsTestBase::RunDatasetOp(
    const DatasetParams& dataset_params,
    std::unique_ptr<OpKernel>* dataset_kernel,
    std::unique_ptr<OpKernelContext::Params>* dataset_ctx_params,
    std::vector<std::unique_ptr<Tensor>>* created_tensors,
    std::unique_ptr<OpKernelContext>* dataset_ctx) {
  std::vector<Tensor*> input_datasets;
  for (auto& input : dataset_params.input_dataset_params()) {
    std::unique_ptr<Tensor> t;
    TF_RETURN_IF_ERROR(MakeDatasetTensor(*input, created_tensors, &t));
    input_datasets.push_back(t.get());
    created_tensors->push_back(std::move(t));
  }
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto input_dataset : input_datasets) {
    inputs.emplace_back(TensorValue(input_dataset));
  }

  // Copy the input tensors, storing them in the `inputs` vectors, and storing
  // owned references to the copies in `created_tensors`.
  for (auto& input : dataset_params.GetInputTensors()) {
    auto copy = absl::make_unique<Tensor>(input);
    inputs.push_back(TensorValue(copy.get()));
    created_tensors->push_back(std::move(copy));
  }

  TF_RETURN_IF_ERROR(MakeDatasetOpKernel(dataset_params, dataset_kernel));
  TF_RETURN_IF_ERROR(CreateDatasetContext(dataset_kernel->get(), &inputs,
                                          dataset_ctx_params, dataset_ctx));
  TF_RETURN_IF_ERROR(RunOpKernel(dataset_kernel->get(), dataset_ctx->get()));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDataset(
    const DatasetParams& dataset_params,
    std::unique_ptr<OpKernel>* dataset_kernel,
    std::unique_ptr<OpKernelContext::Params>* dataset_ctx_params,
    std::unique_ptr<OpKernelContext>* dataset_ctx,
    std::vector<std::unique_ptr<Tensor>>* created_tensors,
    DatasetBase** dataset) {
  TF_RETURN_IF_ERROR(RunDatasetOp(dataset_params, dataset_kernel,
                                  dataset_ctx_params, created_tensors,
                                  dataset_ctx));
  // Assume that DatasetOp has only one output.
  DCHECK_EQ((*dataset_ctx)->num_outputs(), 1);
  TF_RETURN_IF_ERROR(GetDatasetFromContext(dataset_ctx->get(), 0, dataset));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeIterator(
    const DatasetParams& dataset_params, const TestDataset& dataset,
    std::unique_ptr<TestIterator>* iterator) {
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_RETURN_IF_ERROR(
      CreateIteratorContext(dataset.op_kernel_context(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator_base;
  TF_RETURN_IF_ERROR(dataset.dataset()->MakeIterator(
      iterator_ctx.get(), dataset_params.iterator_prefix(), &iterator_base));
  *iterator = std::make_unique<TestIterator>(std::move(iterator_ctx),
                                             std::move(iterator_base));
  return Status::OK();
}

Status DatasetOpsTestBase::RunDatasetOp(const DatasetParams& dataset_params,
                                        std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(RunDatasetOp(dataset_params, &dataset_kernel_, &params_,
                                  &tensors_, &dataset_ctx_));
  for (int i = 0; i < dataset_ctx_->num_outputs(); ++i) {
    outputs->emplace_back(*dataset_ctx_->mutable_output(i));
  }
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDatasetOpKernel(
    const DatasetParams& dataset_params,
    std::unique_ptr<OpKernel>* dataset_kernel) {
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  std::vector<string> input_names;
  TF_RETURN_IF_ERROR(dataset_params.GetInputNames(&input_names));
  AttributeVector attributes;
  TF_RETURN_IF_ERROR(dataset_params.GetAttributes(&attributes));
  NodeDef node_def = test::function::NDef(
      dataset_params.node_name(),
      name_utils::OpName(dataset_params.dataset_type(), params), input_names,
      attributes);
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, dataset_kernel));
  return Status::OK();
}

Status DatasetOpsTestBase::MakeDatasetTensor(
    const DatasetParams& dataset_params,
    std::vector<std::unique_ptr<Tensor>>* created_tensors,
    std::unique_ptr<Tensor>* dataset) {
  // Make sure all the input dataset tensors have been populated.
  std::vector<Tensor*> input_datasets;
  for (auto& input : dataset_params.input_dataset_params()) {
    std::unique_ptr<Tensor> t;
    TF_RETURN_IF_ERROR(MakeDatasetTensor(*input, created_tensors, &t));
    input_datasets.push_back(t.get());
    created_tensors->push_back(std::move(t));
  }

  AttributeVector attributes;
  TF_RETURN_IF_ERROR(dataset_params.GetAttributes(&attributes));

  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto input_dataset : input_datasets) {
    inputs.emplace_back(TensorValue(input_dataset));
  }
  auto input_tensors = dataset_params.GetInputTensors();
  for (auto& input_tensor : input_tensors) {
    inputs.emplace_back(TensorValue(&input_tensor));
  }

  DatasetBase* dataset_base;
  std::unique_ptr<OpKernel> dataset_kernel;
  std::unique_ptr<OpKernelContext::Params> dataset_ctx_params;
  std::unique_ptr<OpKernelContext> dataset_ctx;
  TF_RETURN_IF_ERROR(MakeDatasetOpKernel(dataset_params, &dataset_kernel));
  TF_RETURN_IF_ERROR(CreateDatasetContext(dataset_kernel.get(), &inputs,
                                          &dataset_ctx_params, &dataset_ctx));
  TF_RETURN_IF_ERROR(
      CreateDataset(dataset_kernel.get(), dataset_ctx.get(), &dataset_base));
  Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
  TF_RETURN_IF_ERROR(
      StoreDatasetInVariantTensor(dataset_base, &dataset_tensor));
  *dataset = absl::make_unique<Tensor>(dataset_tensor);
  return Status::OK();
}

DatasetParams::DatasetParams(DataTypeVector output_dtypes,
                             std::vector<PartialTensorShape> output_shapes,
                             string node_name)
    : output_dtypes_(std::move(output_dtypes)),
      output_shapes_(std::move(output_shapes)),
      node_name_(std::move(node_name)) {}

bool DatasetParams::IsDatasetTensor(const Tensor& tensor) {
  return tensor.dtype() == DT_VARIANT &&
         TensorShapeUtils::IsScalar(tensor.shape());
}

RangeDatasetParams::RangeDatasetParams(
    int64 start, int64 stop, int64 step, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name)),
      start_(start),
      stop_(stop),
      step_(step) {}

RangeDatasetParams::RangeDatasetParams(int64 start, int64 stop, int64 step)
    : DatasetParams({DT_INT64}, {PartialTensorShape({})}, "range_dataset"),
      start_(start),
      stop_(stop),
      step_(step) {}

RangeDatasetParams::RangeDatasetParams(int64 start, int64 stop, int64 step,
                                       DataTypeVector output_dtypes)
    : DatasetParams(std::move(output_dtypes), {PartialTensorShape({})},
                    "range_dataset"),
      start_(start),
      stop_(stop),
      step_(step) {}

std::vector<Tensor> RangeDatasetParams::GetInputTensors() const {
  Tensor start_tensor = CreateTensor<int64>(TensorShape({}), {start_});
  Tensor stop_tensor = CreateTensor<int64>(TensorShape({}), {stop_});
  Tensor step_tensor = CreateTensor<int64>(TensorShape({}), {step_});
  return {start_tensor, stop_tensor, step_tensor};
}

Status RangeDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {RangeDatasetOp::kStart, RangeDatasetOp::kStop,
                  RangeDatasetOp::kStep};
  return Status::OK();
}

Status RangeDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{RangeDatasetOp::kOutputTypes, output_dtypes_},
                  {RangeDatasetOp::kOutputShapes, output_shapes_}};
  return Status::OK();
}

string RangeDatasetParams::dataset_type() const {
  return RangeDatasetOp::kDatasetType;
}

std::vector<Tensor> BatchDatasetParams::GetInputTensors() const {
  Tensor batch_size = CreateTensor<int64>(TensorShape({}), {batch_size_});
  Tensor drop_remainder =
      CreateTensor<bool>(TensorShape({}), {drop_remainder_});
  return {batch_size, drop_remainder};
}

Status BatchDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {BatchDatasetOp::kInputDataset, BatchDatasetOp::kBatchSize,
                  BatchDatasetOp::kDropRemainder};
  return Status::OK();
}

Status BatchDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{BatchDatasetOp::kParallelCopy, parallel_copy_},
                  {BatchDatasetOp::kOutputTypes, output_dtypes_},
                  {BatchDatasetOp::kOutputShapes, output_shapes_}};
  return Status::OK();
}

string BatchDatasetParams::dataset_type() const {
  return BatchDatasetOp::kDatasetType;
}

std::vector<Tensor> MapDatasetParams::GetInputTensors() const {
  return other_arguments_;
}

Status MapDatasetParams::GetInputNames(std::vector<string>* input_names) const {
  input_names->emplace_back(MapDatasetOp::kInputDataset);
  for (int i = 0; i < other_arguments_.size(); ++i) {
    input_names->emplace_back(
        absl::StrCat(MapDatasetOp::kOtherArguments, "_", i));
  }
  return Status::OK();
}

Status MapDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {
      {MapDatasetOp::kFunc, func_},
      {MapDatasetOp::kTarguments, type_arguments_},
      {MapDatasetOp::kOutputShapes, output_shapes_},
      {MapDatasetOp::kOutputTypes, output_dtypes_},
      {MapDatasetOp::kUseInterOpParallelism, use_inter_op_parallelism_},
      {MapDatasetOp::kPreserveCardinality, preserve_cardinality_}};
  return Status::OK();
}

string MapDatasetParams::dataset_type() const {
  return MapDatasetOp::kDatasetType;
}

std::vector<FunctionDef> MapDatasetParams::func_lib() const {
  return func_lib_;
}

TensorSliceDatasetParams::TensorSliceDatasetParams(
    std::vector<Tensor> components, string node_name)
    : DatasetParams(TensorSliceDtypes(components),
                    TensorSliceShapes(components), std::move(node_name)),
      components_(std::move(components)) {}

std::vector<Tensor> TensorSliceDatasetParams::GetInputTensors() const {
  return components_;
}

Status TensorSliceDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  input_names->reserve(components_.size());
  for (int i = 0; i < components_.size(); ++i) {
    input_names->emplace_back(
        absl::StrCat(TensorSliceDatasetOp::kComponents, "_", i));
  }
  return Status::OK();
}

Status TensorSliceDatasetParams::GetAttributes(
    AttributeVector* attr_vector) const {
  *attr_vector = {{TensorSliceDatasetOp::kToutputTypes, output_dtypes_},
                  {TensorSliceDatasetOp::kOutputShapes, output_shapes_}};
  return Status::OK();
}

DataTypeVector TensorSliceDatasetParams::TensorSliceDtypes(
    const std::vector<Tensor>& input_components) {
  DataTypeVector dtypes;
  for (const auto& component : input_components) {
    dtypes.emplace_back(component.dtype());
  }
  return dtypes;
}

std::vector<PartialTensorShape> TensorSliceDatasetParams::TensorSliceShapes(
    const std::vector<Tensor>& input_components) {
  std::vector<PartialTensorShape> shapes;
  for (const auto& component : input_components) {
    gtl::InlinedVector<int64, 4> partial_dim_sizes;
    for (int i = 1; i < component.dims(); ++i) {
      partial_dim_sizes.push_back(component.dim_size(i));
    }
    shapes.emplace_back(std::move(partial_dim_sizes));
  }
  return shapes;
}

string TensorSliceDatasetParams::dataset_type() const {
  return TensorSliceDatasetOp::kDatasetType;
}

std::vector<Tensor> TakeDatasetParams::GetInputTensors() const {
  return {CreateTensor<int64>(TensorShape({}), {count_})};
}

Status TakeDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {TakeDatasetOp::kInputDataset, TakeDatasetOp::kCount};
  return Status::OK();
}

Status TakeDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{TakeDatasetOp::kOutputShapes, output_shapes_},
                  {TakeDatasetOp::kOutputTypes, output_dtypes_}};
  return Status::OK();
}

string TakeDatasetParams::dataset_type() const {
  return TakeDatasetOp::kDatasetType;
}

std::vector<Tensor> ConcatenateDatasetParams::GetInputTensors() const {
  return {};
}

Status ConcatenateDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {ConcatenateDatasetOp::kInputDataset,
                  ConcatenateDatasetOp::kAnotherDataset};
  return Status::OK();
}

Status ConcatenateDatasetParams::GetAttributes(
    AttributeVector* attr_vector) const {
  *attr_vector = {{ConcatenateDatasetOp::kOutputTypes, output_dtypes_},
                  {ConcatenateDatasetOp::kOutputShapes, output_shapes_}};
  return Status::OK();
}

string ConcatenateDatasetParams::dataset_type() const {
  return ConcatenateDatasetOp::kDatasetType;
}

}  // namespace data
}  // namespace tensorflow
