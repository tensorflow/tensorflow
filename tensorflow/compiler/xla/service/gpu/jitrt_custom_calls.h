// Copyright 2022 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_

#include <cstdint>
#include <memory>
#include <tuple>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "tensorflow/compiler/xla/mlir/transforms/runtime/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"

namespace xla {
namespace gpu {
class JitRtKernelsCache;
class JitRtGemmConfigCache;
class JitRtCollectiveSupport;
class JitRtAsyncCollectiveSupport;

struct DotDimensionNumbers {
  llvm::ArrayRef<int64_t> lhs_batch;
  llvm::ArrayRef<int64_t> lhs_contract;
  llvm::ArrayRef<int64_t> rhs_batch;
  llvm::ArrayRef<int64_t> rhs_contract;
};

struct ConvDimensionNumbers {
  int64_t input_batch_dim;
  int64_t input_feature_dim;
  llvm::ArrayRef<int64_t> input_spatial_dims;

  int64_t kernel_in_feature_dim;
  int64_t kernel_out_feature_dim;
  llvm::ArrayRef<int64_t> kernel_spatial_dims;

  int64_t output_batch_dim;
  int64_t output_feature_dim;
  llvm::ArrayRef<int64_t> output_spatial_dims;
};

struct ConvBackendConfig {
  int64_t algorithm;
  bool tensor_ops_enabled;
  bool is_cudnn_frontend;
  llvm::ArrayRef<int64_t> knob_ids;
  llvm::ArrayRef<int64_t> knob_values;
  llvm::ArrayRef<int64_t> operand_0_layout;
  llvm::ArrayRef<int64_t> operand_1_layout;
  llvm::ArrayRef<int64_t> result_layout;
  int64_t workspace_size;
};
}  // namespace gpu
}  // namespace xla

namespace xla {
namespace runtime {

using llvm::ArrayRef;

XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(stream_executor::dnn::ActivationMode);
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(stream_executor::fft::Type);
XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(
    stream_executor::cuda::BlasLt::Epilogue);

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::DotDimensionNumbers,
    AggregateMember<ArrayRef<int64_t>>("lhs_batch"),
    AggregateMember<ArrayRef<int64_t>>("lhs_contract"),
    AggregateMember<ArrayRef<int64_t>>("rhs_batch"),
    AggregateMember<ArrayRef<int64_t>>("rhs_contract"));

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::ConvDimensionNumbers,
    // --- input dimensions
    AggregateMember<int64_t>("input_batch_dim"),
    AggregateMember<int64_t>("input_feature_dim"),
    AggregateMember<ArrayRef<int64_t>>("input_spatial_dims"),
    // --- kernel dimensions
    AggregateMember<int64_t>("kernel_in_feature_dim"),
    AggregateMember<int64_t>("kernel_out_feature_dim"),
    AggregateMember<ArrayRef<int64_t>>("kernel_spatial_dims"),
    // --- output dimensions
    AggregateMember<int64_t>("output_batch_dim"),
    AggregateMember<int64_t>("output_feature_dim"),
    AggregateMember<ArrayRef<int64_t>>("output_spatial_dims"));

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::ConvBackendConfig,  //
    AggregateMember<int64_t>("algorithm"),
    AggregateMember<bool>("tensor_ops_enabled"),
    AggregateMember<bool>("is_cudnn_frontend"),
    AggregateMember<ArrayRef<int64_t>>("knob_ids"),
    AggregateMember<ArrayRef<int64_t>>("knob_values"),
    AggregateMember<ArrayRef<int64_t>>("operand_0_layout"),
    AggregateMember<ArrayRef<int64_t>>("operand_1_layout"),
    AggregateMember<ArrayRef<int64_t>>("result_layout"),
    AggregateMember<int64_t>("workspace_size"));

}  // namespace runtime
}  // namespace xla

// Declare explicit dense type ids for all types passed to the custom calls
// as a user data to generate template specializations for fast id lookup.
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::CustomCall,
                                           xla::gpu::JitRtKernelsCache);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::CustomCall,
                                           xla::gpu::JitRtGemmConfigCache);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::CustomCall,
                                           xla::gpu::JitRtCollectiveSupport);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(
    xla::runtime::CustomCall, xla::gpu::JitRtAsyncCollectiveSupport);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(
    xla::runtime::CustomCall, const xla::ServiceExecutableRunOptions);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::CustomCall,
                                           const xla::DebugOptions);
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(xla::runtime::CustomCall,
                                           const std::string);  // ptx
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(
    xla::runtime::CustomCall, const std::vector<uint8_t>);  // cubin
XLA_RUNTIME_DECLARE_EXPLICIT_DENSE_TYPE_ID(
    xla::runtime::CustomCall, se::DeviceMemoryBase);  // temp buffer

namespace xla {
namespace gpu {

// Populate mapping from XLA (SE) enums/structs type id to symbol names.
void PopulateXlaTypeIdNames(runtime::TypeIDNameRegistry& registry);

// Populate encoding from LMHLO attributes to XLA(SE) enums and structs.
void PopulateLmhloToXlaAttrEncoding(
    runtime::CustomCallAttrEncodingSet& encoding);

class JitRtKernelsCache {
 public:
  JitRtKernelsCache() = default;

  ::stream_executor::KernelBase* Get(
      ::stream_executor::StreamExecutor* executor, const char* data,
      llvm::StringRef name);

  ::stream_executor::KernelBase* Set(
      ::stream_executor::StreamExecutor* executor, const char* data,
      llvm::StringRef name,
      std::unique_ptr<::stream_executor::KernelBase> kernel);

 private:
  mutable absl::Mutex mutex_;

  using Key = std::tuple<::stream_executor::StreamExecutor*, const char*,
                         llvm::StringRef>;
  llvm::SmallDenseMap<Key, std::unique_ptr<::stream_executor::KernelBase>>
      kernels_cache_ ABSL_GUARDED_BY(mutex_);
};

class JitRtGemmConfigCache {
 public:
  const GemmConfig* Get(int64_t uid);
  const GemmConfig* Set(int64_t uid, GemmConfig config);

 private:
  mutable absl::Mutex mutex_;

  llvm::SmallDenseMap<int64_t, GemmConfig> configs_ ABSL_GUARDED_BY(mutex_);
};

class JitRtCollectiveSupport {
 public:
  // Maybe block host after the first call to the collective operation with the
  // given uid, to ensure that all devices have allocated the required buffers
  // for their communicators before allowing any device to continue enqueuing
  // operations. Otherwise, the allocations can cause deadlock in the CUDA
  // driver.
  //
  // This basically ports workaround form cr/435058849 to JitRt (see details in
  // the b/215649390).
  Status MaybeBlockAfterFirstRun(int32_t uid, int32_t device_ordinal,
                                 se::Stream* stream);

 private:
  static int64_t Key(int32_t uid, int32_t device_ordinal) {
    return static_cast<int64_t>(uid) << 32 | device_ordinal;
  }

  mutable absl::Mutex mutex_;

  // Store if a particular collective operation was executed at least once. We
  // rely on unique `uid` assigned to each collective operation by the lowering
  // pass.
  llvm::SmallDenseMap<int64_t, bool> executed_ ABSL_GUARDED_BY(mutex_);
};

// Support for running async collective operations communicating via events.
class JitRtAsyncCollectiveSupport {
 public:
  explicit JitRtAsyncCollectiveSupport(se::Stream* async_comm_stream);

  mlir::FailureOr<se::Event> PopEvent(int32_t uid, int32_t device_ordinal);
  mlir::LogicalResult PushEvent(int32_t uid, int32_t device_ordinal,
                                se::Event done_event);

  ::stream_executor::Stream* async_comm_stream() const {
    return async_comm_stream_;
  }

 private:
  static int64_t EventKey(int32_t uid, int32_t device_ordinal) {
    return static_cast<int64_t>(uid) << 32 | device_ordinal;
  }

  mutable absl::Mutex mutex_;

  ::stream_executor::Stream* async_comm_stream_;

  // Store done events for the AllReduceDone to wait on.
  llvm::SmallDenseMap<int64_t, se::Event> done_events_ ABSL_GUARDED_BY(mutex_);
};

xla::runtime::DirectCustomCallLibrary JitRtGpuCustomCalls();

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_JITRT_CUSTOM_CALLS_H_
