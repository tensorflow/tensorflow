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

#include "tensorflow/core/data/dataset_test_base.h"

#include <algorithm>
#include <complex>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tensorflow/tsl/framework/fixedpoint/FixedPoint.h"

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

  return OkStatus();
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
  return OkStatus();
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
  return OkStatus();
}

DatasetOpsTestBase::DatasetOpsTestBase()
    : device_(DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0")),
      device_type_(DEVICE_CPU),
      cpu_num_(kDefaultCPUNum),
      thread_num_(kDefaultThreadNum) {
  allocator_ = device_->GetAllocator(AllocatorAttributes());
}

DatasetOpsTestBase::~DatasetOpsTestBase() {
  if (dataset_) {
    dataset_->Unref();
  }
}

Status DatasetOpsTestBase::ExpectEqual(const Tensor& a, const Tensor& b) {
  if (a.dtype() != b.dtype()) {
    return errors::Internal("Tensor dtypes don't match:\n", a.DebugString(),
                            "\n", b.DebugString());
  }
  switch (a.dtype()) {
#define CASE(DT)                           \
  case DataTypeToEnum<DT>::value:          \
    TF_RETURN_IF_ERROR(IsEqual<DT>(a, b)); \
    break;
    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_tstring(CASE);
#undef CASE
    case DT_VARIANT: {
      if (!TensorShapeUtils::IsScalar(a.shape()) ||
          !TensorShapeUtils::IsScalar(b.shape())) {
        return errors::Internal("Variant tensors must be scalars:\n",
                                a.DebugString(), "\n", b.DebugString());
      }
      const TestVariant* a_object = a.scalar<Variant>()().get<TestVariant>();
      const TestVariant* b_object = b.scalar<Variant>()().get<TestVariant>();
      if (a_object == nullptr || b_object == nullptr) {
        return errors::Internal("Variant types must be `TestVariant`:\n",
                                a.scalar<Variant>()().TypeName(), "\n",
                                b.scalar<Variant>()().TypeName());
      }
      if (*a_object != *b_object) {
        return errors::Internal("Variant tensors aren't equal:\n",
                                a.DebugString(), "\n", b.DebugString());
      }
      break;
    }
    default:
      return errors::Internal("Unsupported dtype: ", a.dtype());
  }
  return OkStatus();
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

  if (produced_tensors.empty()) return OkStatus();
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
  return OkStatus();
}

Status DatasetOpsTestBase::CreateOpKernel(
    const NodeDef& node_def, std::unique_ptr<OpKernel>* op_kernel) {
  OpKernel* kernel;
  Status s;

  std::shared_ptr<const NodeProperties> props;
  TF_RETURN_IF_ERROR(NodeProperties::CreateFromNodeDef(
      node_def, flr_->GetFunctionLibraryDefinition(), &props));
  // Apply attribute defaults.
  auto props_with_defaults = std::make_shared<NodeProperties>(*props);
  for (const auto& attr : props->op_def->attr()) {
    if (attr.has_default_value() &&
        !props->node_def.attr().contains(attr.name())) {
      (*props_with_defaults->node_def.mutable_attr())[attr.name()] =
          attr.default_value();
    }
  }
  TF_RETURN_IF_ERROR(tensorflow::CreateOpKernel(
      device_type_, device_.get(), allocator_, flr_,
      device_->resource_manager(), props_with_defaults, TF_GRAPH_DEF_VERSION,
      &kernel));
  op_kernel->reset(kernel);
  return OkStatus();
}

Status DatasetOpsTestBase::CreateDatasetContext(
    OpKernel* const dateset_kernel,
    gtl::InlinedVector<TensorValue, 4>* const inputs,
    std::unique_ptr<OpKernelContext::Params>* dataset_context_params,
    std::unique_ptr<OpKernelContext>* dataset_context) {
  Status status = CheckOpKernelInput(*dateset_kernel, *inputs);
  if (!status.ok()) {
    VLOG(0) << "WARNING: " << status.ToString();
  }
  TF_RETURN_IF_ERROR(CreateOpKernelContext(
      dateset_kernel, inputs, dataset_context_params, dataset_context));
  return OkStatus();
}

Status DatasetOpsTestBase::CreateDataset(OpKernel* kernel,
                                         OpKernelContext* context,
                                         DatasetBase** const dataset) {
  TF_RETURN_IF_ERROR(RunOpKernel(kernel, context));
  // Assume that DatasetOp has only one output.
  DCHECK_EQ(context->num_outputs(), 1);
  TF_RETURN_IF_ERROR(GetDatasetFromContext(context, 0, dataset));
  return OkStatus();
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
  function_handle_cache_ = std::make_unique<FunctionHandleCache>(flr_);
  params.function_handle_cache = function_handle_cache_.get();
  params.cancellation_manager = cancellation_manager_.get();
  *iterator_context = std::make_unique<IteratorContext>(params);
  return OkStatus();
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
  thread_pool_ = std::make_unique<thread::ThreadPool>(
      Env::Default(), ThreadOptions(), "test_thread_pool", thread_num);
  return OkStatus();
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
  device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));
  resource_mgr_ = std::make_unique<ResourceMgr>("default_container");

  FunctionDefLibrary proto;
  for (const auto& fdef : flib) *(proto.add_function()) = fdef;
  lib_def_ =
      std::make_unique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  OptimizerOptions opts;
  pflr_ = std::make_unique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), /*config=*/nullptr,
      TF_GRAPH_DEF_VERSION, lib_def_.get(), opts, thread_pool_.get(),
      /*parent=*/nullptr,
      /*session_metadata=*/nullptr,
      Rendezvous::Factory{
          [](const int64_t, const DeviceMgr* device_mgr, Rendezvous** r) {
            *r = new IntraProcessRendezvous(device_mgr);
            return OkStatus();
          }});
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  if (thread_pool_ == nullptr) {
    runner_ = [](const std::function<void()>& fn) { fn(); };
  } else {
    runner_ = [this](std::function<void()> fn) {
      thread_pool_->Schedule(std::move(fn));
    };
  }
  return OkStatus();
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
  params.create_kernel = [this, version](
                             const std::shared_ptr<const NodeProperties>& props,
                             OpKernel** kernel) {
    return CreateNonCachedKernel(device_.get(), this->flr_, props, version,
                                 kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
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
  return OkStatus();
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
  auto params = std::make_unique<OpKernelContext::Params>();
  cancellation_manager_ = std::make_unique<CancellationManager>();
  params->cancellation_manager = cancellation_manager_.get();
  params->device = device_.get();
  params->frame_iter = FrameAndIter(0, 0);
  params->function_library = flr_;
  params->inputs = *inputs;
  params->op_kernel = kernel;
  params->resource_manager = resource_mgr_.get();
  params->runner = &runner_;
  slice_reader_cache_ =
      std::make_unique<checkpoint::TensorSliceReaderCacheWrapper>();
  params->slice_reader_cache = slice_reader_cache_.get();
  step_container_ =
      std::make_unique<ScopedStepContainer>(0, [](const string&) {});
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

  *context = std::make_unique<OpKernelContext>(params.get());
  *context_params = std::move(params);
  return OkStatus();
}

Status DatasetOpsTestBase::CreateSerializationContext(
    std::unique_ptr<SerializationContext>* context) {
  *context =
      std::make_unique<SerializationContext>(SerializationContext::Params{});
  return OkStatus();
}

Status DatasetOpsTestBase::CheckOpKernelInput(
    const OpKernel& kernel, const gtl::InlinedVector<TensorValue, 4>& inputs) {
  if (kernel.num_inputs() != inputs.size()) {
    return errors::InvalidArgument("The number of input elements should be ",
                                   kernel.num_inputs(),
                                   ", but got: ", inputs.size());
  }
  return OkStatus();
}

Status DatasetOpsTestBase::AddDatasetInput(
    gtl::InlinedVector<TensorValue, 4>* inputs, DataTypeVector input_types,
    DataType dtype, const TensorShape& shape) {
  if (input_types.size() < inputs->size()) {
    return errors::InvalidArgument("Adding more inputs than types: ",
                                   inputs->size(), " vs. ", input_types.size());
  }
  bool is_ref = IsRefType(input_types[inputs->size()]);
  auto input = std::make_unique<Tensor>(allocator_, dtype, shape);

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

  return OkStatus();
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
  // Call GetNext one more time to make sure it still reports
  // end_of_sequence = True.
  std::vector<Tensor> unused;
  TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &unused, &end_of_sequence));
  EXPECT_TRUE(end_of_sequence);

  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order=*/compare_order));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckIteratorSkip(
    int num_to_skip, int expected_num_skipped, bool get_next,
    const std::vector<Tensor>& expected_outputs, bool compare_order) {
  IteratorBase* iterator = iterator_.get();
  IteratorContext* ctx = iterator_ctx_.get();

  bool end_of_sequence = false;
  int num_skipped = 0;
  TF_RETURN_IF_ERROR(
      iterator->Skip(ctx, num_to_skip, &end_of_sequence, &num_skipped));
  EXPECT_TRUE(num_skipped == expected_num_skipped);
  if (get_next) {
    EXPECT_TRUE(!end_of_sequence);
    std::vector<Tensor> out_tensors;
    TF_RETURN_IF_ERROR(iterator->GetNext(ctx, &out_tensors, &end_of_sequence));
    TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                             /*compare_order=*/compare_order));
  }
  return OkStatus();
}

Status DatasetOpsTestBase::CheckSplitProviderFullIteration(
    const DatasetParams& params, const std::vector<Tensor>& expected_outputs) {
  std::unique_ptr<TestDataset> dataset;
  TF_RETURN_IF_ERROR(MakeDataset(params, &dataset));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(dataset->dataset()->MakeSplitProviders(&split_providers));
  std::unique_ptr<TestIterator> iterator;
  TF_RETURN_IF_ERROR(
      MakeIterator(params, *dataset, std::move(split_providers), &iterator));
  TF_RETURN_IF_ERROR(CheckIteratorGetNext(iterator.get(), expected_outputs,
                                          /*compare_order=*/true));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckSplitProviderShardedIteration(
    const DatasetParams& params, int64_t num_shards, int64_t shard_index,
    const std::vector<Tensor>& expected_outputs) {
  std::unique_ptr<TestDataset> dataset;
  TF_RETURN_IF_ERROR(MakeDataset(params, &dataset));
  std::vector<std::unique_ptr<SplitProvider>> split_providers;
  TF_RETURN_IF_ERROR(dataset->dataset()->MakeSplitProviders(&split_providers));
  for (int i = 0; i < split_providers.size(); ++i) {
    split_providers[i] = std::make_unique<ShardingSplitProvider>(
        num_shards, shard_index, std::move(split_providers[i]));
  }
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_RETURN_IF_ERROR(
      CreateIteratorContext(dataset->op_kernel_context(), &iterator_ctx));
  IteratorContext::Params iterator_params(iterator_ctx.get());
  std::move(split_providers.begin(), split_providers.end(),
            std::back_inserter(iterator_params.split_providers));
  iterator_ctx = std::make_unique<IteratorContext>(iterator_params);
  int mid_breakpoint = expected_outputs.size() / 2;
  int near_end_breakpoint = expected_outputs.size() - 1;
  int end_breakpoint = expected_outputs.size();
  TF_RETURN_IF_ERROR(CheckIteratorSaveAndRestore(
      dataset->dataset(), iterator_ctx.get(), params.iterator_prefix(),
      expected_outputs,
      /*breakpoints=*/
      {0, mid_breakpoint, near_end_breakpoint, end_breakpoint},
      /*compare_order=*/true));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckDatasetNodeName(
    const string& expected_dataset_node_name) {
  EXPECT_EQ(dataset_->node_name(), expected_dataset_node_name);
  return OkStatus();
}

Status DatasetOpsTestBase::CheckDatasetTypeString(
    const string& expected_type_str) {
  EXPECT_EQ(dataset_->type_string(), expected_type_str);
  return OkStatus();
}

Status DatasetOpsTestBase::CheckDatasetOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
  TF_EXPECT_OK(
      VerifyTypesMatch(dataset_->output_dtypes(), expected_output_dtypes));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckDatasetOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(dataset_->output_shapes(),
                                      expected_output_shapes));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckDatasetCardinality(int expected_cardinality) {
  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
  return OkStatus();
}

Status DatasetOpsTestBase::CheckDatasetOptions(
    const Options& expected_options) {
  EXPECT_EQ(dataset_->options().SerializeAsString(),
            expected_options.SerializeAsString());
  return OkStatus();
}

Status DatasetOpsTestBase::CheckIteratorOutputDtypes(
    const DataTypeVector& expected_output_dtypes) {
  TF_EXPECT_OK(
      VerifyTypesMatch(iterator_->output_dtypes(), expected_output_dtypes));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckIteratorOutputShapes(
    const std::vector<PartialTensorShape>& expected_output_shapes) {
  TF_EXPECT_OK(VerifyShapesCompatible(iterator_->output_shapes(),
                                      expected_output_shapes));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckIteratorPrefix(
    const string& expected_iterator_prefix) {
  EXPECT_EQ(iterator_->prefix(), expected_iterator_prefix);
  return OkStatus();
}

Status DatasetOpsTestBase::CheckIteratorSaveAndRestore(
    DatasetBase* dataset, IteratorContext* iterator_ctx,
    const std::string& iterator_prefix,
    const std::vector<Tensor>& expected_outputs,
    const std::vector<int>& breakpoints, bool compare_order) {
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(dataset->MakeIterator(iterator_ctx, /*parent=*/nullptr,
                                           iterator_prefix, &iterator));
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
    TF_EXPECT_OK(RestoreIterator(iterator_ctx, &reader, iterator_prefix,
                                 *dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_RETURN_IF_ERROR(
          iterator->GetNext(iterator_ctx, &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order=*/compare_order));
  return OkStatus();
}

Status DatasetOpsTestBase::CheckIteratorSaveAndRestore(
    const std::string& iterator_prefix,
    const std::vector<Tensor>& expected_outputs,
    const std::vector<int>& breakpoints, bool compare_order) {
  return CheckIteratorSaveAndRestore(dataset_, iterator_ctx_.get(),
                                     iterator_prefix, expected_outputs,
                                     breakpoints, compare_order);
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
  TF_RETURN_IF_ERROR(
      dataset_->MakeIterator(iterator_ctx_.get(), /*parent=*/nullptr,
                             dataset_params.iterator_prefix(), &iterator_));
  initialized_ = true;
  return OkStatus();
}

Status DatasetOpsTestBase::InitializeRuntime(
    const DatasetParams& dataset_params) {
  TF_RETURN_IF_ERROR(InitThreadPool(thread_num_));
  TF_RETURN_IF_ERROR(
      InitFunctionLibraryRuntime(dataset_params.func_lib(), cpu_num_));
  return OkStatus();
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
  return OkStatus();
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
  inputs.reserve(input_datasets.size());
  for (auto input_dataset : input_datasets) {
    inputs.emplace_back(TensorValue(input_dataset));
  }

  // Copy the input tensors, storing them in the `inputs` vectors, and storing
  // owned references to the copies in `created_tensors`.
  for (auto& input : dataset_params.GetInputTensors()) {
    auto copy = std::make_unique<Tensor>(input);
    inputs.push_back(TensorValue(copy.get()));
    created_tensors->push_back(std::move(copy));
  }

  TF_RETURN_IF_ERROR(MakeDatasetOpKernel(dataset_params, dataset_kernel));
  TF_RETURN_IF_ERROR(CreateDatasetContext(dataset_kernel->get(), &inputs,
                                          dataset_ctx_params, dataset_ctx));
  TF_RETURN_IF_ERROR(RunOpKernel(dataset_kernel->get(), dataset_ctx->get()));
  return OkStatus();
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
  return OkStatus();
}

Status DatasetOpsTestBase::MakeIterator(
    const DatasetParams& dataset_params, const TestDataset& dataset,
    std::vector<std::unique_ptr<SplitProvider>> split_providers,
    std::unique_ptr<TestIterator>* iterator) {
  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_RETURN_IF_ERROR(
      CreateIteratorContext(dataset.op_kernel_context(), &iterator_ctx));
  IteratorContext::Params iterator_params(iterator_ctx.get());
  std::move(split_providers.begin(), split_providers.end(),
            std::back_inserter(iterator_params.split_providers));

  iterator_ctx = std::make_unique<IteratorContext>(iterator_params);
  std::unique_ptr<IteratorBase> iterator_base;
  TF_RETURN_IF_ERROR(dataset.dataset()->MakeIterator(
      iterator_ctx.get(), /*parent=*/nullptr, dataset_params.iterator_prefix(),
      &iterator_base));
  *iterator = std::make_unique<TestIterator>(std::move(iterator_ctx),
                                             std::move(iterator_base));
  return OkStatus();
}

Status DatasetOpsTestBase::MakeIterator(
    const DatasetParams& dataset_params, const TestDataset& dataset,
    std::unique_ptr<TestIterator>* iterator) {
  return MakeIterator(dataset_params, dataset, /*split_providers=*/{},
                      iterator);
}

Status DatasetOpsTestBase::RunDatasetOp(const DatasetParams& dataset_params,
                                        std::vector<Tensor>* outputs) {
  TF_RETURN_IF_ERROR(RunDatasetOp(dataset_params, &dataset_kernel_, &params_,
                                  &tensors_, &dataset_ctx_));
  for (int i = 0; i < dataset_ctx_->num_outputs(); ++i) {
    outputs->emplace_back(*dataset_ctx_->mutable_output(i));
  }
  return OkStatus();
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
  NodeDef node_def =
      test::function::NDef(dataset_params.node_name(), dataset_params.op_name(),
                           input_names, attributes);
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, dataset_kernel));
  return OkStatus();
}

Status DatasetOpsTestBase::MakeGetOptionsOpKernel(
    const DatasetParams& dataset_params, std::unique_ptr<OpKernel>* op_kernel) {
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  std::vector<string> input_names;
  TF_RETURN_IF_ERROR(dataset_params.GetInputNames(&input_names));
  AttributeVector attributes;
  TF_RETURN_IF_ERROR(dataset_params.GetAttributes(&attributes));
  NodeDef node_def = test::function::NDef(dataset_params.node_name(),
                                          dataset_params.dataset_type(),
                                          input_names, attributes);
  TF_RETURN_IF_ERROR(CreateOpKernel(node_def, op_kernel));
  return OkStatus();
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

  auto input_tensors = dataset_params.GetInputTensors();
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.reserve(input_datasets.size() + input_tensors.size());
  for (auto input_dataset : input_datasets) {
    inputs.emplace_back(TensorValue(input_dataset));
  }
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
  *dataset = std::make_unique<Tensor>(dataset_tensor);
  return OkStatus();
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
    int64_t start, int64_t stop, int64_t step, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name)),
      start_(start),
      stop_(stop),
      step_(step) {}

RangeDatasetParams::RangeDatasetParams(int64_t start, int64_t stop,
                                       int64_t step)
    : DatasetParams({DT_INT64}, {PartialTensorShape({})}, "range_dataset"),
      start_(start),
      stop_(stop),
      step_(step) {}

RangeDatasetParams::RangeDatasetParams(int64_t start, int64_t stop,
                                       int64_t step,
                                       DataTypeVector output_dtypes)
    : DatasetParams(std::move(output_dtypes), {PartialTensorShape({})},
                    "range_dataset"),
      start_(start),
      stop_(stop),
      step_(step) {}

std::vector<Tensor> RangeDatasetParams::GetInputTensors() const {
  Tensor start_tensor = CreateTensor<int64_t>(TensorShape({}), {start_});
  Tensor stop_tensor = CreateTensor<int64_t>(TensorShape({}), {stop_});
  Tensor step_tensor = CreateTensor<int64_t>(TensorShape({}), {step_});
  return {start_tensor, stop_tensor, step_tensor};
}

Status RangeDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {"start", "stop", "step"};
  return OkStatus();
}

Status RangeDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{"output_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"replicate_on_split", false},
                  {"metadata", ""}};
  return OkStatus();
}

string RangeDatasetParams::dataset_type() const { return "Range"; }

std::vector<Tensor> BatchDatasetParams::GetInputTensors() const {
  Tensor batch_size = CreateTensor<int64_t>(TensorShape({}), {batch_size_});
  Tensor drop_remainder =
      CreateTensor<bool>(TensorShape({}), {drop_remainder_});
  return {batch_size, drop_remainder};
}

Status BatchDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {"input_dataset", "batch_size", "drop_remainder"};
  return OkStatus();
}

Status BatchDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{"parallel_copy", parallel_copy_},
                  {"output_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"metadata", ""}};
  return OkStatus();
}

string BatchDatasetParams::dataset_type() const { return "Batch"; }

std::vector<Tensor> MapDatasetParams::GetInputTensors() const {
  return other_arguments_;
}

Status MapDatasetParams::GetInputNames(std::vector<string>* input_names) const {
  input_names->emplace_back("input_dataset");
  for (int i = 0; i < other_arguments_.size(); ++i) {
    input_names->emplace_back(absl::StrCat("other_arguments_", i));
  }
  return OkStatus();
}

Status MapDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{"f", func_},
                  {"Targuments", type_arguments_},
                  {"output_shapes", output_shapes_},
                  {"output_types", output_dtypes_},
                  {"use_inter_op_parallelism", use_inter_op_parallelism_},
                  {"preserve_cardinality", preserve_cardinality_},
                  {"metadata", ""}};
  return OkStatus();
}

string MapDatasetParams::dataset_type() const { return "Map"; }

std::vector<FunctionDef> MapDatasetParams::func_lib() const {
  return func_lib_;
}

TensorSliceDatasetParams::TensorSliceDatasetParams(
    std::vector<Tensor> components, string node_name, bool is_files)
    : DatasetParams(TensorSliceDtypes(components),
                    TensorSliceShapes(components), std::move(node_name)),
      components_(std::move(components)),
      is_files_(is_files) {}

std::vector<Tensor> TensorSliceDatasetParams::GetInputTensors() const {
  return components_;
}

Status TensorSliceDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  input_names->reserve(components_.size());
  for (int i = 0; i < components_.size(); ++i) {
    input_names->emplace_back(absl::StrCat("components_", i));
  }
  return OkStatus();
}

Status TensorSliceDatasetParams::GetAttributes(
    AttributeVector* attr_vector) const {
  *attr_vector = {{"Toutput_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"is_files", is_files_},
                  {"replicate_on_split", false},
                  {"metadata", ""}};
  return OkStatus();
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
    gtl::InlinedVector<int64_t, 4> partial_dim_sizes;
    for (int i = 1; i < component.dims(); ++i) {
      partial_dim_sizes.push_back(component.dim_size(i));
    }
    shapes.emplace_back(std::move(partial_dim_sizes));
  }
  return shapes;
}

string TensorSliceDatasetParams::dataset_type() const { return "TensorSlice"; }

std::vector<Tensor> TakeDatasetParams::GetInputTensors() const {
  return {CreateTensor<int64_t>(TensorShape({}), {count_})};
}

Status TakeDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {"input_dataset", "count"};
  return OkStatus();
}

Status TakeDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{"output_shapes", output_shapes_},
                  {"output_types", output_dtypes_},
                  {"metadata", ""}};
  return OkStatus();
}

string TakeDatasetParams::dataset_type() const { return "Take"; }

std::vector<Tensor> ConcatenateDatasetParams::GetInputTensors() const {
  return {};
}

Status ConcatenateDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  *input_names = {"input_dataset", "another_dataset"};
  return OkStatus();
}

Status ConcatenateDatasetParams::GetAttributes(
    AttributeVector* attr_vector) const {
  *attr_vector = {{"output_types", output_dtypes_},
                  {"output_shapes", output_shapes_},
                  {"metadata", ""}};
  return OkStatus();
}

string ConcatenateDatasetParams::dataset_type() const { return "Concatenate"; }

std::vector<Tensor> OptionsDatasetParams::GetInputTensors() const { return {}; }

Status OptionsDatasetParams::GetInputNames(
    std::vector<string>* input_names) const {
  input_names->emplace_back("input_dataset");
  return OkStatus();
}

Status OptionsDatasetParams::GetAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{"serialized_options", serialized_options_},
                  {"output_shapes", output_shapes_},
                  {"output_types", output_dtypes_},
                  {"metadata", ""}};
  return OkStatus();
}

string OptionsDatasetParams::dataset_type() const { return "Options"; }

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(
    DatasetOpsTestBase::TestVariant,
    DatasetOpsTestBase::TestVariant::kTypeName);

}  // namespace data
}  // namespace tensorflow
