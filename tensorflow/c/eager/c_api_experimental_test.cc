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
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/cc/profiler/profiler.h"
#include "tensorflow/core/lib/monitoring/collection_registry.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/profiler/trace_events.pb.h"

using tensorflow::string;

namespace tensorflow {
namespace {

static bool HasSubstr(absl::string_view base, absl::string_view substr) {
  bool ok = str_util::StrContains(base, substr);
  EXPECT_TRUE(ok) << base << ", expected substring " << substr;
  return ok;
}

void ExecuteWithProfiling(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  TFE_ProfilerContext* profiler_context = TFE_NewProfilerContext();
  TFE_ProfilerContextSetEagerContext(profiler_context, ctx);
  TFE_Profiler* profiler = TFE_NewProfiler(profiler_context);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);
  TFE_DeleteProfilerContext(profiler_context);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m, m);
  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;

  // Run op on GPU if it is present.
  string gpu_device_name;
  if (GetDeviceName(ctx, &gpu_device_name, "GPU")) {
    TFE_OpSetDevice(matmul, gpu_device_name.c_str(), status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    const char* device_name = TFE_OpGetDevice(matmul, status);
    ASSERT_TRUE(strstr(device_name, "GPU:0") != nullptr);
  }

  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);

  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);
  TF_Buffer* profiler_result = TF_NewBuffer();
  TFE_ProfilerSerializeToString(ctx, profiler, profiler_result, status);
  TFE_DeleteProfiler(profiler);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  profiler::Trace profile_proto;
  EXPECT_TRUE(profile_proto.ParseFromString(
      {reinterpret_cast<const char*>(profiler_result->data),
       profiler_result->length}));
  string profile_proto_str = profile_proto.DebugString();
  if (!gpu_device_name.empty()) {
    EXPECT_TRUE(HasSubstr(profile_proto_str, "/device:GPU:0"));
    // device name with "stream:all" is collected by Device Tracer.
    EXPECT_TRUE(HasSubstr(profile_proto_str, "stream:all"));
    // TODO(fishx): move following check out from this if statement.
    // This is collected by TraceMe
    EXPECT_TRUE(HasSubstr(profile_proto_str, "/host:CPU"));
  }
  EXPECT_TRUE(HasSubstr(profile_proto_str, "/device:CPU:0"));
  EXPECT_TRUE(HasSubstr(profile_proto_str, "MatMul"));
  TF_DeleteBuffer(profiler_result);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  TFE_DeleteTensorHandle(retvals[0]);
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
TEST(CAPI, ExecuteWithTracing) { ExecuteWithProfiling(false); }
TEST(CAPI, ExecuteWithTracingAsync) { ExecuteWithProfiling(true); }

TEST(CAPI, MultipleProfilerSession) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(false));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ProfilerContext* profiler_context = TFE_NewProfilerContext();
  TFE_ProfilerContextSetEagerContext(profiler_context, ctx);

  TFE_Profiler* profiler1 = TFE_NewProfiler(profiler_context);
  EXPECT_TRUE(TFE_ProfilerIsOk(profiler1));

  TFE_Profiler* profiler2 = TFE_NewProfiler(profiler_context);
  EXPECT_FALSE(TFE_ProfilerIsOk(profiler2));

  TFE_DeleteProfiler(profiler1);
  TFE_DeleteProfiler(profiler2);
  TFE_DeleteProfilerContext(profiler_context);
}

TEST(CAPI, MonitoringSetGauge) {
  TFE_MonitoringSetGauge("test/gauge", "label", 1);
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<monitoring::CollectedMetrics> metrics =
      collection_registry->CollectMetrics(options);

  EXPECT_EQ("test/gauge", metrics->point_set_map.at("test/gauge")->metric_name);
  EXPECT_EQ(1,
            metrics->point_set_map.at("test/gauge")->points.at(0)->int64_value);

  TFE_MonitoringSetGauge("test/gauge", "label", 5);
  metrics = collection_registry->CollectMetrics(options);
  EXPECT_EQ(5,
            metrics->point_set_map.at("test/gauge")->points.at(0)->int64_value);
}

TEST(CAPI, MonitoringAddCounter) {
  TFE_MonitoringAddCounter("test/counter", "label", 1);
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<monitoring::CollectedMetrics> metrics =
      collection_registry->CollectMetrics(options);

  EXPECT_EQ("test/counter",
            metrics->point_set_map.at("test/counter")->metric_name);
  EXPECT_EQ(
      1, metrics->point_set_map.at("test/counter")->points.at(0)->int64_value);

  TFE_MonitoringAddCounter("test/counter", "label", 5);
  metrics = collection_registry->CollectMetrics(options);
  EXPECT_EQ(
      6, metrics->point_set_map.at("test/counter")->points.at(0)->int64_value);
}

TEST(CAPI, MonitoringAddSampler) {
  TFE_MonitoringAddSampler("test/sampler", "label", 1.0);
  auto* collection_registry = monitoring::CollectionRegistry::Default();
  monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<monitoring::CollectedMetrics> metrics =
      collection_registry->CollectMetrics(options);

  EXPECT_EQ("test/sampler",
            metrics->point_set_map.at("test/sampler")->metric_name);
  EXPECT_EQ(1.0, metrics->point_set_map.at("test/sampler")
                     ->points.at(0)
                     ->histogram_value.sum());

  TFE_MonitoringAddSampler("test/sampler", "label", 5.0);
  metrics = collection_registry->CollectMetrics(options);
  EXPECT_EQ(6.0, metrics->point_set_map.at("test/sampler")
                     ->points.at(0)
                     ->histogram_value.sum());
}

}  // namespace
}  // namespace tensorflow
