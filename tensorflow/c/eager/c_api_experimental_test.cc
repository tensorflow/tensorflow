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

#include "tensorflow/c/eager/c_api_experimental.h"

#include <string.h>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

using tensorflow::string;

namespace tensorflow {
namespace {

static bool HasSubstr(absl::string_view base, absl::string_view substr) {
  bool ok = absl::StrContains(base, substr);
  EXPECT_TRUE(ok) << base << ", expected substring " << substr;
  return ok;
}

TEST(CAPI, MonitoringCounter0) {
  TF_Status* status = TF_NewStatus();
  auto* counter =
      TFE_MonitoringNewCounter0("test/counter", status, "description");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  auto* cell = TFE_MonitoringGetCellCounter0(counter);
  TFE_MonitoringCounterCellIncrementBy(cell, 1);
  EXPECT_EQ(TFE_MonitoringCounterCellValue(cell), 1);
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<monitoring::CollectedMetrics> metrics =
      collection_registry->CollectMetrics(options);

  EXPECT_EQ("test/counter",
            metrics->point_set_map.at("test/counter")->metric_name);
  EXPECT_EQ(
      1, metrics->point_set_map.at("test/counter")->points.at(0)->int64_value);

  TFE_MonitoringCounterCellIncrementBy(cell, 5);
  EXPECT_EQ(TFE_MonitoringCounterCellValue(cell), 6);
  metrics = collection_registry->CollectMetrics(options);
  EXPECT_EQ(
      6, metrics->point_set_map.at("test/counter")->points.at(0)->int64_value);

  TFE_MonitoringDeleteCounter0(counter);
  metrics = collection_registry->CollectMetrics(options);
  EXPECT_EQ(metrics->point_set_map.end(),
            metrics->point_set_map.find("test/counter"));
}

TEST(CAPI, MonitoringCounterMultiple) {
  TF_Status* status = TF_NewStatus();
  auto* counter1 = TFE_MonitoringNewCounter1("test/counter1", status,
                                             "description", "label1");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* cell1 = TFE_MonitoringGetCellCounter1(counter1, "test");
  TFE_MonitoringCounterCellIncrementBy(cell1, 1);
  EXPECT_EQ(TFE_MonitoringCounterCellValue(cell1), 1);

  auto* counter2 = TFE_MonitoringNewCounter2("test/counter2", status,
                                             "description", "label1", "label2");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  auto* cell2 = TFE_MonitoringGetCellCounter2(counter2, "foo", "bar");
  TFE_MonitoringCounterCellIncrementBy(cell2, 2);
  EXPECT_EQ(TFE_MonitoringCounterCellValue(cell2), 2);

  TFE_MonitoringDeleteCounter1(counter1);
  TFE_MonitoringDeleteCounter2(counter2);
}

TEST(CAPI, MonitoringGauge0) {
  TF_Status* status = TF_NewStatus();
  auto* gauge = TFE_MonitoringNewIntGauge0("test/gauge", status, "test");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* cell = TFE_MonitoringGetCellIntGauge0(gauge);
  TFE_MonitoringIntGaugeCellSet(cell, 1);
  EXPECT_EQ(TFE_MonitoringIntGaugeCellValue(cell), 1);
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<monitoring::CollectedMetrics> metrics =
      collection_registry->CollectMetrics(options);

  EXPECT_EQ("test/gauge", metrics->point_set_map.at("test/gauge")->metric_name);
  EXPECT_EQ(1,
            metrics->point_set_map.at("test/gauge")->points.at(0)->int64_value);

  TFE_MonitoringIntGaugeCellSet(cell, 5);
  metrics = collection_registry->CollectMetrics(options);
  EXPECT_EQ(5,
            metrics->point_set_map.at("test/gauge")->points.at(0)->int64_value);
  TFE_MonitoringDeleteIntGauge0(gauge);
  TF_DeleteStatus(status);
}

TEST(CAPI, MonitoringMultipleGauge) {
  TF_Status* status = TF_NewStatus();
  auto* gauge1 =
      TFE_MonitoringNewBoolGauge1("test/gauge1", status, "test", "label1");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* cell1 = TFE_MonitoringGetCellBoolGauge1(gauge1, "foo");
  TFE_MonitoringBoolGaugeCellSet(cell1, true);
  EXPECT_TRUE(TFE_MonitoringBoolGaugeCellValue(cell1));
  TFE_MonitoringDeleteBoolGauge1(gauge1);

  auto* gauge2 = TFE_MonitoringNewStringGauge2("test/gauge2", status, "test",
                                               "label1", "label2");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* cell2 = TFE_MonitoringGetCellStringGauge2(gauge2, "foo", "bar");
  TFE_MonitoringStringGaugeCellSet(cell2, "str");
  auto* buf = new TF_Buffer;
  TFE_MonitoringStringGaugeCellValue(cell2, buf);
  string data(static_cast<const char*>(buf->data), buf->length);
  TF_DeleteBuffer(buf);
  EXPECT_EQ(data, "str");
  TFE_MonitoringDeleteStringGauge2(gauge2);
  TF_DeleteStatus(status);
}

TEST(CAPI, MonitoringSampler0) {
  TF_Status* status = TF_NewStatus();
  auto* buckets = TFE_MonitoringNewExponentialBuckets(1.0, 2.0, 2);
  auto* sampler =
      TFE_MonitoringNewSampler0("test/sampler", buckets, status, "test");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* cell = TFE_MonitoringGetCellSampler0(sampler);
  TFE_MonitoringSamplerCellAdd(cell, 1.0);
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<monitoring::CollectedMetrics> metrics =
      collection_registry->CollectMetrics(options);

  EXPECT_EQ("test/sampler",
            metrics->point_set_map.at("test/sampler")->metric_name);
  EXPECT_EQ(1.0, metrics->point_set_map.at("test/sampler")
                     ->points.at(0)
                     ->histogram_value.sum());

  TFE_MonitoringSamplerCellAdd(cell, 5.0);
  metrics = collection_registry->CollectMetrics(options);
  EXPECT_EQ(6.0, metrics->point_set_map.at("test/sampler")
                     ->points.at(0)
                     ->histogram_value.sum());
  TFE_MonitoringDeleteBuckets(buckets);
  TFE_MonitoringDeleteSampler0(sampler);
  TF_DeleteStatus(status);
}

TEST(CAPI, MonitoringMultipleSampler) {
  TF_Status* status = TF_NewStatus();
  auto* buckets = TFE_MonitoringNewExponentialBuckets(1.0, 2.0, 2);
  auto* sampler1 = TFE_MonitoringNewSampler1("test/sampler1", buckets, status,
                                             "test", "label1");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* cell1 = TFE_MonitoringGetCellSampler1(sampler1, "foo");
  TFE_MonitoringSamplerCellAdd(cell1, 1.0);
  TFE_MonitoringSamplerCellAdd(cell1, 2.0);
  TF_Buffer* result1 = TF_NewBuffer();
  TFE_MonitoringSamplerCellValue(cell1, result1);
  tensorflow::HistogramProto histogram1;
  EXPECT_TRUE(histogram1.ParseFromString(
      {reinterpret_cast<const char*>(result1->data), result1->length}));
  EXPECT_EQ(histogram1.sum(), 3.0);
  TF_DeleteBuffer(result1);
  TFE_MonitoringDeleteSampler1(sampler1);

  auto* sampler2 = TFE_MonitoringNewSampler2("test/sampler2", buckets, status,
                                             "test", "label1", "label2");
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  auto* cell2 = TFE_MonitoringGetCellSampler2(sampler2, "foo", "bar");
  TFE_MonitoringSamplerCellAdd(cell2, 2.0);
  TFE_MonitoringSamplerCellAdd(cell2, 3.0);
  TF_Buffer* result2 = TF_NewBuffer();
  TFE_MonitoringSamplerCellValue(cell2, result2);
  tensorflow::HistogramProto histogram2;
  EXPECT_TRUE(histogram2.ParseFromString(
      {reinterpret_cast<const char*>(result2->data), result2->length}));
  EXPECT_EQ(histogram2.sum(), 5.0);
  TF_DeleteBuffer(result2);
  TFE_MonitoringDeleteSampler2(sampler2);

  TFE_MonitoringDeleteBuckets(buckets);
  TF_DeleteStatus(status);
}

TEST(CAPI, CancellationManager) {
  TFE_CancellationManager* c_mgr = TFE_NewCancellationManager();
  EXPECT_FALSE(TFE_CancellationManagerIsCancelled(c_mgr));
  TFE_CancellationManagerStartCancel(c_mgr);
  EXPECT_TRUE(TFE_CancellationManagerIsCancelled(c_mgr));
  TFE_DeleteCancellationManager(c_mgr);
}

TEST(CAPI, ExecutorContextDestructionOrder) {
  TF_Status* status = TF_NewStatus();

  {
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_Context* ctx = TFE_NewContext(opts, status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    TFE_DeleteContextOptions(opts);
    TFE_Executor* executor = TFE_NewExecutor(
        /*is_async=*/false, /*enable_streaming_enqueue=*/true);
    TFE_ContextSetExecutorForThread(ctx, executor);

    TFE_DeleteContext(ctx);
    TFE_DeleteExecutor(executor);
  }

  {
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_Context* ctx = TFE_NewContext(opts, status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    TFE_DeleteContextOptions(opts);
    TFE_Executor* executor = TFE_NewExecutor(
        /*is_async=*/false, /*enable_streaming_enqueue=*/true);
    TFE_ContextSetExecutorForThread(ctx, executor);

    TFE_DeleteExecutor(executor);
    TFE_DeleteContext(ctx);
  }
  TF_DeleteStatus(status);
}

TEST(CAPI, Function_ident_CPU) {
  // First create a simple identity function.
  TF_Graph* function_graph = TF_NewGraph();
  TF_OperationDescription* arg_descr =
      TF_NewOperation(function_graph, "Placeholder", "arg");
  TF_SetAttrType(arg_descr, "dtype", TF_INT32);
  TF_Status* status = TF_NewStatus();
  TF_Operation* arg = TF_FinishOperation(arg_descr, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_OperationDescription* id_descr =
      TF_NewOperation(function_graph, "Identity", "id");
  TF_SetAttrType(id_descr, "T", TF_INT32);
  TF_AddInput(id_descr, {arg, 0});
  TF_Operation* id = TF_FinishOperation(id_descr, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_Output input{arg, 0};
  TF_Output output{id, 0};
  TF_Function* fn =
      TF_GraphToFunction(function_graph, "ident", 0, 1, &id, 1, &input, 1,
                         &output, nullptr, nullptr, "test", status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_DeleteGraph(function_graph);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TFE_DeleteContextOptions(opts);
  TFE_ContextAddFunction(ctx, fn, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_DeleteFunction(fn);

  for (bool async : {false, true, false}) {
    TFE_Executor* old_executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_Executor* executor = TFE_NewExecutor(
        /*is_async=*/async, /*enable_streaming_enqueue=*/true);
    TFE_ContextSetExecutorForThread(ctx, executor);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TF_Tensor* t =
        TF_AllocateTensor(TF_INT32, nullptr, 0, 1 * sizeof(tensorflow::int32));
    *reinterpret_cast<tensorflow::int32*>(TF_TensorData(t)) = 42;
    TFE_TensorHandle* h = TFE_NewTensorHandle(t, status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    TF_DeleteTensor(t);

    TFE_Op* op = TFE_NewOp(ctx, "ident", status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    TFE_OpAddInput(op, h, status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);

    std::vector<TFE_TensorHandle*> result;
    result.push_back(nullptr);
    int num_retvals = 1;
    TFE_Execute(op, result.data(), &num_retvals, status);
    TFE_DeleteOp(op);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    ASSERT_EQ(num_retvals, 1);

    TF_Tensor* r = TFE_TensorHandleResolve(result[0], status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    EXPECT_EQ(*reinterpret_cast<tensorflow::int32*>(TF_TensorData(r)), 42);
    TFE_ContextSetExecutorForThread(ctx, old_executor);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);
    TFE_DeleteExecutor(old_executor);
    TFE_DeleteTensorHandle(h);
    TF_DeleteTensor(r);
    TFE_DeleteTensorHandle(result[0]);
  }
  TFE_ContextRemoveFunction(ctx, "ident", status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TFE_DeleteContext(ctx);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_DeleteStatus(status);
}

void Executor_MatMul_CPU(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_Executor* old_executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_Executor* executor = TFE_NewExecutor(
      /*is_async=*/async, /*enable_streaming_enqueue=*/true);
  TFE_ContextSetExecutorForThread(ctx, executor);

  TFE_TensorHandle* m = TestMatrixTensorHandle(ctx);
  TFE_Op* matmul = MatMulOp(ctx, m, m);
  TFE_TensorHandle* retvals[2] = {nullptr, nullptr};
  int num_retvals = 2;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(1, num_retvals);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_ContextSetExecutorForThread(ctx, old_executor);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteExecutor(executor);
  TFE_DeleteExecutor(old_executor);
  TFE_DeleteContext(ctx);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float product[4] = {0};
  EXPECT_EQ(sizeof(product), TF_TensorByteSize(t));
  memcpy(&product[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(7, product[0]);
  EXPECT_EQ(10, product[1]);
  EXPECT_EQ(15, product[2]);
  EXPECT_EQ(22, product[3]);
  TF_DeleteStatus(status);
}
TEST(CAPI, Executor_MatMul_CPU) { Executor_MatMul_CPU(false); }
TEST(CAPI, Executor_MatMul_CPUAsync) { Executor_MatMul_CPU(true); }

void Deleter(void* data, size_t unused, void* tensor_handle) {
  TFE_DeleteTensorHandle(static_cast<TFE_TensorHandle*>(tensor_handle));
}

TEST(CAPI, TensorHandleOnDeviceMemory) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle(ctx);
  TF_Tensor* m_data = TFE_TensorHandleResolve(m, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float* m_float = static_cast<float*>(TF_TensorData(m_data));
  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_devices = TF_DeviceListCount(devices);
  for (int d = 0; d < num_devices; ++d) {
    const char* name = TF_DeviceListName(devices, d, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* copy = TFE_TensorHandleCopyToDevice(m, ctx, name, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    void* data = TFE_TensorHandleDevicePointer(copy, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    size_t size = TFE_TensorHandleDeviceMemorySize(copy, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    int64_t dims[] = {2, 2};
    TFE_TensorHandle* copy_aliased = TFE_NewTensorHandleFromDeviceMemory(
        ctx, name, TF_FLOAT, dims, 2, data, size, &Deleter, copy, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* on_host =
        TFE_TensorHandleCopyToDevice(copy_aliased, ctx, "CPU:0", status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TF_Tensor* resolved = TFE_TensorHandleResolve(on_host, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    const float* resolved_data =
        static_cast<const float*>(TF_TensorData(resolved));
    EXPECT_EQ(0, memcmp(m_float, resolved_data, 4 * sizeof(float)));
    TF_DeleteTensor(resolved);
    TFE_DeleteTensorHandle(copy_aliased);  // Note that this will delete copy.
    TFE_DeleteTensorHandle(on_host);
  }
  TF_DeleteDeviceList(devices);
  TF_DeleteTensor(m_data);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}

TEST(CAPI, TensorHandleNullptr) {
  TFE_TensorHandle* h = nullptr;
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);

  const char* device_type = TFE_TensorHandleDeviceType(h, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
  ASSERT_EQ(device_type, nullptr);
  ASSERT_EQ("Invalid handle", string(TF_Message(status.get())));

  TF_SetStatus(status.get(), TF_OK, "");

  int device_id = TFE_TensorHandleDeviceID(h, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
  ASSERT_EQ(device_id, -1);
  ASSERT_EQ("Invalid handle", string(TF_Message(status.get())));
}

TEST(CAPI, TensorHandleDevices) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* hcpu = TestMatrixTensorHandle(ctx);
  const char* device_type = TFE_TensorHandleDeviceType(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_TRUE(absl::StrContains(device_type, "CPU")) << device_type;
  int device_id = TFE_TensorHandleDeviceID(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_EQ(0, device_id) << device_id;

  // Disable the test if no GPU is present.
  string gpu_device_name;
  if (GetDeviceName(ctx, &gpu_device_name, "GPU")) {
    TFE_TensorHandle* hgpu = TFE_TensorHandleCopyToDevice(
        hcpu, ctx, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    TFE_Op* shape_op = ShapeOp(ctx, hgpu);
    TFE_OpSetDevice(shape_op, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(shape_op, &retvals[0], &num_retvals, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    device_type = TFE_TensorHandleDeviceType(retvals[0], status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    ASSERT_TRUE(absl::StrContains(device_type, "GPU")) << device_type;

    device_id = TFE_TensorHandleDeviceID(retvals[0], status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    ASSERT_EQ(0, device_id) << device_id;

    TFE_DeleteOp(shape_op);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteTensorHandle(hgpu);
  }

  TFE_DeleteTensorHandle(hcpu);
  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TensorHandleDefaults) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* h_default = TestMatrixTensorHandle(ctx);
  const char* device_type = TFE_TensorHandleDeviceType(h_default, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_TRUE(absl::StrContains(device_type, "CPU")) << device_type;
  int device_id = TFE_TensorHandleDeviceID(h_default, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_EQ(0, device_id) << device_id;

  TFE_TensorHandle* h_cpu = TFE_TensorHandleCopyToDevice(
      h_default, ctx, "/device:CPU:0", status.get());
  const char* device_type_cpu = TFE_TensorHandleDeviceType(h_cpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_TRUE(absl::StrContains(device_type_cpu, "CPU")) << device_type_cpu;
  int device_id_cpu = TFE_TensorHandleDeviceID(h_cpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_EQ(0, device_id_cpu) << device_id_cpu;

  TFE_DeleteTensorHandle(h_default);
  TFE_DeleteTensorHandle(h_cpu);
  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);
}

}  // namespace
}  // namespace tensorflow
