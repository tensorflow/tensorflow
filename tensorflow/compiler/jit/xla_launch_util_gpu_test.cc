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

// TODO(b/282059652): This test currently only runs inside Google because PJRT
// in GPU device is not buildable in open-source environment. Resolve the issue
// and enable the test in open-source.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/pjrt_device_context.h"
#include "tensorflow/compiler/jit/pjrt_tensor_buffer.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
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
    auto jit_device_type = DeviceType(DEVICE_GPU_XLA_JIT);
    rollout_config.AllowForDeviceInXlaLaunch(device_type);
    rollout_config.AllowForDeviceInXlaCompileOnDemand(device_type);

    SessionOptions session_options;
    auto* virtual_devices = session_options.config.mutable_gpu_options()
                                ->mutable_experimental()
                                ->add_virtual_devices();
    virtual_devices->add_memory_limit_mb(256);
    virtual_devices->add_memory_limit_mb(256);
    (*session_options.config
          .mutable_device_count())[device_type.type_string()] = 2;
    std::vector<std::unique_ptr<Device>> devices;
    CHECK_OK(DeviceFactory::GetFactory(device_type.type_string())
                 ->CreateDevices(session_options,
                                 "/job:localhost/replica:0/task:0", &devices));
    CHECK_GE(devices.size(), 2);
    device_ = devices[0].get();
    secondary_device_ = std::move(devices[1]);
    SetDevice(device_type, std::move(devices[0]));

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
    CHECK_OK(pjrt_client_or.status());
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
  std::unique_ptr<Device> secondary_device_;
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

  auto pjrt_device = pjrt_client_->LookupAddressableDevice(
      xla::LocalDeviceId(device_->parsed_name().id));
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

TEST_F(PjRtExecutionUtilGpuTest,
       RunPjRtExecutableWithPjRtTensorBufferFromDifferentVirtualGpu) {
  XlaOpRegistry::RegisterCompilationKernels();
  TF_ASSERT_OK(NodeDefBuilder("AddV2", "AddV2")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_INT32)
                   .Device("/job:localhost/replica:0/task:0/device:GPU:0")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Create device tensors on virtual GPU 0 (cuda:0), which are backed by
  // PjRtTensorBuffer with device() == cuda:0.
  Tensor* a = CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {1, 2, 3});
  Tensor* b = CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {4, 5, 6});
  inputs_.push_back({nullptr, a});
  inputs_.push_back({nullptr, b});

  CreateContext();

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({1, 3});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({1, 3});

  const XlaCompiler::CompilationResult* result;
  xla::PjRtLoadedExecutable* executable;
  CompileToExecutable(args, &result, &executable);

  // Execute on virtual GPU 1 (cuda:1) using the PjRtTensorBuffer tensors
  // created on virtual GPU 0 (cuda:0).
  TF_ASSERT_OK_AND_ASSIGN(
      xla::PjRtDevice * pjrt_device_1,
      pjrt_client_->LookupAddressableDevice(xla::LocalDeviceId(1)));
  std::vector<const Tensor*> inputs = {a, b};
  const DeviceType& device_type = GetDeviceType(context_.get());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<xla::PjRtBuffer>> execute_outputs,
      RunPjRtExecutable(/*num_missing_prefix_ctx_inputs=*/0, inputs,
                        /*variable_snapshots=*/{}, /*updated_variables=*/{},
                        device_type, /*use_pjrt_tensor_buffer=*/true, *result,
                        pjrt_device_1, pjrt_client_, executable));

  ASSERT_EQ(execute_outputs.size(), 1);
  EXPECT_EQ(execute_outputs[0]->device(), pjrt_device_1);
  std::shared_ptr<xla::Literal> literal = *execute_outputs[0]->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal, xla::LiteralUtil::CreateR2<int32_t>({{5, 7, 9}})));
}

TEST_F(PjRtExecutionUtilGpuTest,
       PreparePjRtExecutableArgumentsPersistsPjRtTensorBuffer) {
  // Allocate a raw GPU tensor without a PjRtTensorBuffer.
  Tensor raw_gpu_tensor(device_allocator_, DT_INT32, TensorShape({1, 3}));
  Tensor* host_tensor =
      CreateHostTensor<int32_t>(TensorShape({1, 3}), {7, 8, 9});
  se::Stream* stream = device_->tensorflow_accelerator_device_info()->stream;
  stream_executor::DeviceAddressBase dst_mem(
      const_cast<char*>(raw_gpu_tensor.tensor_data().data()),
      raw_gpu_tensor.TotalBytes());
  TF_ASSERT_OK(stream->Memcpy(&dst_mem, host_tensor->tensor_data().data(),
                              raw_gpu_tensor.TotalBytes()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_EQ(
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(&raw_gpu_tensor)),
      nullptr);

  std::vector<const Tensor*> inputs = {&raw_gpu_tensor};
  std::vector<int> input_mapping = {0};
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtDevice * pjrt_device,
                          pjrt_client_->LookupAddressableDevice(
                              xla::LocalDeviceId(device_->parsed_name().id)));

  std::vector<xla::PjRtBuffer*> exec_args_1;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> owned_args_1;
  absl::flat_hash_set<int> non_donatable_1;
  TF_EXPECT_OK(PreparePjRtExecutableArguments(
      /*num_missing_prefix_ctx_inputs=*/0, input_mapping, inputs,
      /*variable_snapshots=*/{}, pjrt_client_, pjrt_device,
      /*use_pjrt_tensor_buffer=*/true, &exec_args_1, &owned_args_1,
      &non_donatable_1));

  ASSERT_EQ(exec_args_1.size(), 1);
  EXPECT_TRUE(owned_args_1.empty());
  EXPECT_TRUE(non_donatable_1.contains(0));
  const auto* pjrt_tensor_buffer =
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(&raw_gpu_tensor));
  ASSERT_NE(pjrt_tensor_buffer, nullptr);
  EXPECT_EQ(exec_args_1[0], pjrt_tensor_buffer->pjrt_buffer());

  // A subsequent call reuses the persisted PjRtBuffer.
  std::vector<xla::PjRtBuffer*> exec_args_2;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> owned_args_2;
  absl::flat_hash_set<int> non_donatable_2;
  TF_EXPECT_OK(PreparePjRtExecutableArguments(
      /*num_missing_prefix_ctx_inputs=*/0, input_mapping, inputs,
      /*variable_snapshots=*/{}, pjrt_client_, pjrt_device,
      /*use_pjrt_tensor_buffer=*/true, &exec_args_2, &owned_args_2,
      &non_donatable_2));
  ASSERT_EQ(exec_args_2.size(), 1);
  EXPECT_TRUE(owned_args_2.empty());
  EXPECT_TRUE(non_donatable_2.contains(0));
  EXPECT_EQ(exec_args_2[0], exec_args_1[0]);
}

TEST_F(
    PjRtExecutionUtilGpuTest,
    LockVariablesAndPreparePjRtExecutableArgumentsForReadOnlyRawGpuVariable) {
  // Create an initialized GPU Var backed by a raw GPU tensor (without a
  // PjRtTensorBuffer).
  auto* var = new Var(DT_INT32);
  *var->tensor() = Tensor(device_allocator_, DT_INT32, TensorShape({1, 3}));
  var->is_initialized = true;
  core::ScopedUnref var_ref(var);
  ASSERT_EQ(
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(var->tensor())),
      nullptr);

  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtDevice * pjrt_device,
                          pjrt_client_->LookupAddressableDevice(
                              xla::LocalDeviceId(device_->parsed_name().id)));

  // Without `upgrade_read_only_for_pjrt_tensor_buffer`, LockVariables keeps the
  // shared reader lock even though the variable's TensorBuffer is not a
  // PjRtTensorBuffer.
  {
    var->Ref();
    std::vector<VariableInfo> variables;
    variables.emplace_back(/*index=*/0, "read_only_var", var);
    variables[0].set_read_only();
    TF_ASSERT_OK(LockVariables(absl::MakeSpan(variables)));
    EXPECT_FALSE(variables[0].lock_held());
    EXPECT_TRUE(variables[0].shared_lock_held());
  }

  // First PJRT call: LockVariables upgrades the read-only variable's lock from
  // shared to exclusive when `upgrade_read_only_for_pjrt_tensor_buffer` is true
  // because its TensorBuffer is not yet a PjRtTensorBuffer, and
  // PreparePjRtExecutableArguments persists the PjRtTensorBuffer in-place even
  // when another Tensor (e.g., SnapshotResourceVariables) shares the underlying
  // buffer.
  {
    var->Ref();
    std::vector<VariableInfo> variables;
    variables.emplace_back(/*index=*/0, "read_only_var", var);
    variables[0].set_read_only();
    TF_ASSERT_OK(
        LockVariables(absl::MakeSpan(variables),
                      /*upgrade_read_only_for_pjrt_tensor_buffer=*/true));
    EXPECT_TRUE(variables[0].lock_held());
    EXPECT_FALSE(variables[0].shared_lock_held());

    Tensor snapshot_reference = *var->tensor();
    ASSERT_FALSE(var->tensor()->RefCountIsOne());

    std::vector<xla::PjRtBuffer*> exec_args;
    std::vector<std::unique_ptr<xla::PjRtBuffer>> owned_args;
    absl::flat_hash_set<int> non_donatable;
    TF_ASSERT_OK(PreparePjRtExecutableArguments(
        /*num_missing_prefix_ctx_inputs=*/0, /*input_mapping=*/{0},
        /*inputs=*/{}, GetVariableSnapshots(variables), pjrt_client_,
        pjrt_device, /*use_pjrt_tensor_buffer=*/true, &exec_args, &owned_args,
        &non_donatable));

    ASSERT_EQ(exec_args.size(), 1);
    EXPECT_TRUE(owned_args.empty());
    EXPECT_TRUE(non_donatable.contains(0));
    const auto* pjrt_tensor_buffer =
        dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(var->tensor()));
    ASSERT_NE(pjrt_tensor_buffer, nullptr);
    EXPECT_EQ(exec_args[0], pjrt_tensor_buffer->pjrt_buffer());
  }

  // Subsequent call: now that the variable's Tensor is backed by a
  // PjRtTensorBuffer, LockVariables keeps the shared reader lock without
  // upgrading to an exclusive lock.
  {
    var->Ref();
    std::vector<VariableInfo> variables;
    variables.emplace_back(/*index=*/0, "read_only_var", var);
    variables[0].set_read_only();
    TF_ASSERT_OK(
        LockVariables(absl::MakeSpan(variables),
                      /*upgrade_read_only_for_pjrt_tensor_buffer=*/true));
    EXPECT_FALSE(variables[0].lock_held());
    EXPECT_TRUE(variables[0].shared_lock_held());
  }
}

TEST_F(PjRtExecutionUtilGpuTest, CopyCPUTensorToGPUProducesPjRtTensorBuffer) {
  Tensor* host_tensor =
      CreateHostTensor<int32_t>(TensorShape({1, 3}), {10, 20, 30});
  Tensor gpu_tensor(device_allocator_, DT_INT32, TensorShape({1, 3}));
  ASSERT_EQ(
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(&gpu_tensor)),
      nullptr);

  DeviceContext* default_context =
      device_->tensorflow_accelerator_device_info()->default_context;
  TF_ASSERT_OK(default_context->CopyCPUTensorToDeviceSync(host_tensor, device_,
                                                          &gpu_tensor));

  const auto* pjrt_tensor_buffer =
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(&gpu_tensor));
  ASSERT_NE(pjrt_tensor_buffer, nullptr);
  ASSERT_NE(pjrt_tensor_buffer->pjrt_buffer(), nullptr);
  std::shared_ptr<xla::Literal> literal =
      *pjrt_tensor_buffer->pjrt_buffer()->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal, xla::LiteralUtil::CreateR2<int32_t>({{10, 20, 30}})));
}

TEST_F(PjRtExecutionUtilGpuTest,
       PreparePjRtExecutableArgumentsUnwrapsFullTensorSubSlice) {
  Tensor* host_tensor =
      CreateHostTensor<int32_t>(TensorShape({1, 3}), {10, 20, 30});
  Tensor gpu_tensor(DT_INT32, TensorShape({1, 3}));
  TF_ASSERT_OK(device_context_->CopyCPUTensorToDeviceSync(host_tensor, device_,
                                                          &gpu_tensor));
  const auto* root_pjrt_tensor_buffer =
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(&gpu_tensor));
  ASSERT_NE(root_pjrt_tensor_buffer, nullptr);

  // SubSlice(0) wraps the PjRtTensorBuffer in a SubBuffer<int32_t>.
  Tensor sub_sliced = gpu_tensor.SubSlice(0);
  Tensor rebatched;
  ASSERT_TRUE(rebatched.CopyFrom(sub_sliced, TensorShape({1, 3})));
  ASSERT_EQ(
      dynamic_cast<const PjRtTensorBuffer*>(DMAHelper::buffer(&rebatched)),
      nullptr);

  std::vector<const Tensor*> inputs = {&rebatched};
  std::vector<int> input_mapping = {0};
  TF_ASSERT_OK_AND_ASSIGN(xla::PjRtDevice * pjrt_device,
                          pjrt_client_->LookupAddressableDevice(
                              xla::LocalDeviceId(device_->parsed_name().id)));

  std::vector<xla::PjRtBuffer*> exec_args;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> owned_args;
  absl::flat_hash_set<int> non_donatable;
  TF_EXPECT_OK(PreparePjRtExecutableArguments(
      /*num_missing_prefix_ctx_inputs=*/0, input_mapping, inputs,
      /*variable_snapshots=*/{}, pjrt_client_, pjrt_device,
      /*use_pjrt_tensor_buffer=*/true, &exec_args, &owned_args,
      &non_donatable));

  ASSERT_EQ(exec_args.size(), 1);
  EXPECT_TRUE(owned_args.empty());
  EXPECT_TRUE(non_donatable.contains(0));
  EXPECT_EQ(exec_args[0], root_pjrt_tensor_buffer->pjrt_buffer());
}

}  // namespace
}  // namespace tensorflow
