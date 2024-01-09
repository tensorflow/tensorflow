/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// TODO(b/282059652): This test currently only runs inside Google because PJRT
// in GPU device is not buildable in open-source environment. Resolve the issue
// and enable the test in open-source.
#if defined(PLATFORM_GOOGLE)

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/pjrt_device_context.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tests/literal_test_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"
#include "tsl/framework/allocator.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {

using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;
using PjRtDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::PjRtLoadedExecutable, xla::PjRtClient>;

absl::flat_hash_map<int, const Tensor*> GetVariableSnapshots(
    const std::vector<VariableInfo>& variables) {
  absl::flat_hash_map<int, const Tensor*> variable_snapshots;
  for (int i = 0; i < variables.size(); i++) {
    variable_snapshots[variables[i].index()] = variables[i].var()->tensor();
  }
  return variable_snapshots;
}

class PjRtExecutionUtilGpuTest : public OpsTestBase {
 public:
  PjRtExecutionUtilGpuTest() {
    // Set flag to use PJRT for device compilation and execution.
    auto& rollout_config = GetXlaOpsCommonFlags()->tf_xla_use_device_api;
    rollout_config.enabled_for_xla_launch_ = true;
    rollout_config.enabled_for_compile_on_demand_ = true;
    rollout_config.enabled_for_gpu_ = true;

    // Set flag to enable using XLA devices. PJRT currently is only supported
    // for XLA devices.
    GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;

    // Add and setup the GPU device.
    auto device_type = DeviceType(DEVICE_GPU);
    auto jit_device_type = DeviceType(DEVICE_GPU);
    rollout_config.AllowForDeviceInXlaLaunch(device_type);
    rollout_config.AllowForDeviceInXlaCompileOnDemand(device_type);

    auto device =
        DeviceFactory::NewDevice(device_type.type_string(), SessionOptions(),
                                 "/job:localhost/replica:0/task:0");
    device_ = device.get();
    SetDevice(device_type, std::move(device));

    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_fns{
        UseNoPreferenceLayoutFn(), IdentityShapeRepresentationFn()};
    device_context_ = core::RefCountPtr<DeviceContext>(
        new PjRtDeviceContext(shape_fns, /*use_pjrt_tensor_buffer=*/true));

    // Get the host allocator.
    AllocatorAttributes host_alloc_attr;
    host_alloc_attr.set_on_host(true);
    host_allocator_ = device_->GetAllocator(host_alloc_attr);

    AllocatorAttributes device_alloc_attr;
    device_alloc_attr.set_on_host(false);
    device_allocator_ = device_->GetAllocator(device_alloc_attr);

    // Create the DeviceCompiler to help with compiling executables.
    auto pjrt_client_or = GetOrCreatePjRtClient(device_type_);
    TF_CHECK_OK(pjrt_client_or.status());
    pjrt_client_ = pjrt_client_or.value();
    device_compiler_ = new PjRtDeviceCompiler(
        std::make_unique<PjRtDeviceExecutablePersistor>(
            PjRtDeviceExecutablePersistor::Config(), jit_device_type),
        std::make_unique<PjRtDeviceCompilerClient>(pjrt_client_));
    profiler_ = new DeviceCompilationProfiler();

    compiler_options_.device_type = jit_device_type;
    compiler_options_.client = nullptr;
    compiler_options_.flib_def = flib_def_.get();
  }

  ~PjRtExecutionUtilGpuTest() override {
    for (const auto& tensor : tensors_) {
      delete tensor;
    }
    tensors_.clear();
    device_context_->Unref();
    core::ScopedUnref device_compiler_ref(device_compiler_);
    core::ScopedUnref profiler_ref(profiler_);
  }

  // Creates a Tensor on host using the host_allocator_
  template <typename T>
  Tensor* CreateHostTensor(const TensorShape& shape,
                           const gtl::ArraySlice<T> data) {
    Tensor* host_tensor =
        new Tensor(host_allocator_, DataTypeToEnum<T>::v(), shape);
    test::FillValues<T>(host_tensor, data);
    tensors_.push_back(host_tensor);
    return host_tensor;
  }

  // Creates a Tensor on device using the device_allocator_
  template <typename T>
  Tensor* CreateDeviceTensor(const TensorShape& shape,
                             const gtl::ArraySlice<T> data) {
    Tensor* host_tensor = CreateHostTensor<T>(shape, data);
    Tensor* device_tensor =
        new Tensor(device_allocator_, DataTypeToEnum<T>::v(), shape);
    TF_EXPECT_OK(device_context_->CopyCPUTensorToDeviceSync(
        host_tensor, device_, device_tensor));

    tensors_.push_back(device_tensor);
    return device_tensor;
  }

  // Compiles the op set in the context_ to a PjRtLoadedExecutable
  void CompileToExecutable(const std::vector<XlaCompiler::Argument>& args,
                           const XlaCompiler::CompilationResult** result,
                           xla::PjRtLoadedExecutable** executable,
                           XlaCompiler::CompileOptions compile_options = {}) {
    TF_EXPECT_OK(device_compiler_->CompileSingleOpIfNeeded(
        compiler_options_, args, compile_options, context_.get(), profiler_,
        result, executable));
  }
  // Creates a Variable. Doesn't add it to the resource manager.
  template <typename T>
  Var* CreateVariable(const string& name, const TensorShape& shape,
                      const gtl::ArraySlice<T> data) {
    Tensor* init_var_value = CreateDeviceTensor<T>(shape, data);
    Var* var = new Var(DataTypeToEnum<T>::v());
    *var->tensor() = *init_var_value;
    var->is_initialized = true;

    return var;
  }

  // Creates a Variable, adds it to the resource manager and also adds it as one
  // of the inputs in the context_
  template <typename T>
  void AddVariableInput(const string& name, const TensorShape& shape,
                        const gtl::ArraySlice<T> data) {
    Var* var = CreateVariable<T>(name, shape, data);
    ResourceMgr* rm = device_->resource_manager();
    TF_ASSERT_OK(rm->Create(rm->default_container(), name, var));

    ResourceHandle handle;
    handle.set_device(device_->name());
    handle.set_container(rm->default_container());
    handle.set_name(name);
    TypeIndex type_index = TypeIndex::Make<Var>();
    handle.set_hash_code(type_index.hash_code());
    handle.set_maybe_type_name(type_index.name());

    Tensor* input = new Tensor(host_allocator_, DT_RESOURCE, TensorShape({}));
    input->scalar<ResourceHandle>()() = handle;
    tensors_.push_back(input);
    inputs_.push_back({nullptr, input});
  }

 protected:
  core::RefCountPtr<DeviceContext> device_context_;
  Allocator* host_allocator_;
  Allocator* device_allocator_;

  XlaCompiler::Options compiler_options_;
  xla::PjRtClient* pjrt_client_;
  PjRtDeviceCompiler* device_compiler_;
  DeviceCompilationProfiler* profiler_;
};

TEST_F(PjRtExecutionUtilGpuTest, PreparePjRtExecutableArguments) {
  std::vector<const Tensor*> inputs;
  inputs.push_back(CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {0, 0, 0}));
  inputs.push_back(CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {1, 2, 3}));
  inputs.push_back(CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {4, 5, 6}));
  int num_missing_prefix_ctx_inputs = 2;
  std::vector<int> input_mapping{3, 4};
  std::vector<VariableInfo> variables;

  auto pjrt_device =
      pjrt_client_->LookupAddressableDevice(device_->parsed_name().id);
  TF_EXPECT_OK(pjrt_device.status());

  std::vector<xla::PjRtBuffer*> exec_args;
  exec_args.reserve(input_mapping.size());
  absl::flat_hash_set<int> non_donatable_input_indices;
  TF_EXPECT_OK(PreparePjRtExecutableArguments(
      num_missing_prefix_ctx_inputs, input_mapping, inputs,
      GetVariableSnapshots(variables),
      /*pjrt_client=*/pjrt_client_, /*pjrt_device=*/*pjrt_device,
      /*use_pjrt_tensor_buffer=*/true, &exec_args,
      /*owned_args=*/{}, &non_donatable_input_indices));

  EXPECT_EQ(exec_args.size(), 2);

  std::shared_ptr<xla::Literal> literal1 = *exec_args[0]->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal1, xla::LiteralUtil::CreateR2<int32_t>({{1, 2, 3}})));

  std::shared_ptr<xla::Literal> literal2 = *exec_args[1]->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal2, xla::LiteralUtil::CreateR2<int32_t>({{4, 5, 6}})));
}

}  // namespace
}  // namespace tensorflow

#endif  // PLATFORM_GOOGLE
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
