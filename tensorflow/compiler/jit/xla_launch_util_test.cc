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

#include "tensorflow/compiler/jit/xla_launch_util.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/compiler/jit/device_compiler.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/variable_info.h"
#include "tensorflow/compiler/jit/variable_info_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
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
#include "tensorflow/tsl/framework/allocator.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace {
using PjRtDeviceCompiler =
    DeviceCompiler<xla::PjRtLoadedExecutable, xla::PjRtClient>;
using PjRtDeviceExecutablePersistor =
    DeviceExecutablePersistor<xla::PjRtLoadedExecutable, xla::PjRtClient>;

class PjRtExecutionUtilTest : public OpsTestBase {
 public:
  PjRtExecutionUtilTest() {
    // Set flag to use PJRT for device compilation and execution.
    GetXlaOpsCommonFlags()->tf_xla_use_device_api = true;

    // Set flag to enable using XLA devices. PJRT currently is only supported
    // for XLA devices.
    GetXlaDeviceFlags()->tf_xla_enable_xla_devices = true;

    // Add and setup the XLA_CPU device.
    auto device_type = DeviceType(DEVICE_XLA_CPU);
    auto jit_device_type = DeviceType(DEVICE_CPU_XLA_JIT);
    auto device =
        DeviceFactory::NewDevice(device_type.type_string(), SessionOptions(),
                                 "/job:localhost/replica:0/task:0");
    device_ = device.get();
    SetDevice(device_type, std::move(device));

    // Create PjRtClient for XLA_CPU.
    TF_CHECK_OK(SetPjRtClientInTFGlobalResourceManager(
        device_type,
        xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1)
            .value()));

    // device_context_ should be a PjRtDeviceContext.
    TF_CHECK_OK(device_->TryGetDeviceContext(&device_context_));

    // Get the host allocator.
    AllocatorAttributes host_alloc_attr;
    host_alloc_attr.set_on_host(true);
    host_allocator_ = device_->GetAllocator(host_alloc_attr);

    // Get the device allocator. This should give us an AsyncValueAllocator.
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

  ~PjRtExecutionUtilTest() override {
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

  // Gets the `output_index`-th output set in the context_
  Tensor* GetOutput(int output_index) {
    CHECK_LT(output_index, context_->num_outputs());
    Tensor* device_tensor = context_->mutable_output(output_index);
    managed_outputs_.resize(context_->num_outputs());
    if (managed_outputs_[output_index]) {
      return managed_outputs_[output_index];
    }

    Tensor* host_tensor = new Tensor(host_allocator_, device_tensor->dtype(),
                                     device_tensor->shape());
    TF_EXPECT_OK(device_context_->CopyDeviceTensorToCPUSync(
        device_tensor, "", device_, host_tensor));
    managed_outputs_[output_index] = host_tensor;
    return host_tensor;
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

  // Runs a PjRtLoadedExecutable with the given inputs, variables. Requires the
  // XlaCompiler::CompilationResult that was used to build the executable.
  StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> RunExecutable(
      const std::vector<const Tensor*>& inputs,
      const std::vector<VariableInfo>& variables,
      const XlaCompiler::CompilationResult* result,
      xla::PjRtLoadedExecutable* executable) {
    TF_ASSIGN_OR_RETURN(auto pjrt_device, pjrt_client_->LookupAddressableDevice(
                                              device_->parsed_name().id));

    const std::vector<xla::PjRtBuffer*> executable_args =
        PreparePjRtExecutableArguments(result->input_mapping, inputs,
                                       variables);

    xla::ExecuteOptions exe_options;
    exe_options.arguments_are_tupled = false;
    exe_options.untuple_result = true;

    // TODO(b/257548614): currently PJRT is compiled as portable (num_replica =
    // 1 and num_partition = 1). Support multiple partitions case.
    return executable->ExecutePortable(executable_args, pjrt_device,
                                       exe_options);
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
  DeviceContext* device_context_;
  Allocator* host_allocator_;
  Allocator* device_allocator_;

  XlaCompiler::Options compiler_options_;
  xla::PjRtClient* pjrt_client_;
  PjRtDeviceCompiler* device_compiler_;
  DeviceCompilationProfiler* profiler_;
};

TEST_F(PjRtExecutionUtilTest, PreparePjRtExecutableArguments) {
  std::vector<const Tensor*> inputs;
  inputs.push_back(CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {0, 0, 0}));
  inputs.push_back(CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {1, 2, 3}));
  inputs.push_back(CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {4, 5, 6}));
  std::vector<int> input_mapping{1, 2};

  std::vector<xla::PjRtBuffer*> exec_args =
      PreparePjRtExecutableArguments(input_mapping, inputs, {});

  EXPECT_EQ(exec_args.size(), 2);

  std::shared_ptr<xla::Literal> literal1 = *exec_args[0]->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal1, xla::LiteralUtil::CreateR2<int32_t>({{1, 2, 3}})));

  std::shared_ptr<xla::Literal> literal2 = *exec_args[1]->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal2, xla::LiteralUtil::CreateR2<int32_t>({{4, 5, 6}})));
}

TEST_F(PjRtExecutionUtilTest, PreparePjRtExecutableArgumentsVariableInputs) {
  std::vector<VariableInfo> variables;
  Var* var1 = CreateVariable<int32>("v1", TensorShape({1, 2}), {1, 2});
  variables.emplace_back(1, "v1", var1);
  Var* var2 = CreateVariable<int32>("v2", TensorShape({1, 2}), {3, 4});
  variables.emplace_back(2, "v2", var2);

  std::vector<const Tensor*> inputs;
  inputs.push_back(CreateDeviceTensor<int32_t>(TensorShape({1, 3}), {0, 0, 0}));
  std::vector<int> input_mapping{1, 2};

  std::vector<xla::PjRtBuffer*> exec_args =
      PreparePjRtExecutableArguments(input_mapping, inputs, variables);

  EXPECT_EQ(exec_args.size(), 2);

  std::shared_ptr<xla::Literal> literal1 = *exec_args[0]->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal1, xla::LiteralUtil::CreateR2<int32_t>({{1, 2}})));

  std::shared_ptr<xla::Literal> literal2 = *exec_args[1]->ToLiteralSync();
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(
      *literal2, xla::LiteralUtil::CreateR2<int32_t>({{3, 4}})));
}

TEST_F(PjRtExecutionUtilTest, PopulateCtxOutputs) {
  XlaOpRegistry::RegisterCompilationKernels();
  TF_EXPECT_OK(NodeDefBuilder("AddV2", "AddV2")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_INT32)
                   .Device("/job:localhost/replica:0/task:0/device:XLA_CPU:0")
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  // Add inputs.
  Tensor* a = CreateDeviceTensor<int32>(TensorShape({1, 3}), {1, 2, 3});
  Tensor* b = CreateDeviceTensor<int32>(TensorShape({1, 3}), {4, 5, 6});
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

  std::vector<const Tensor*> inputs;
  inputs.push_back(a);
  inputs.push_back(b);
  TF_ASSERT_OK_AND_ASSIGN(auto execute_outputs,
                          RunExecutable(inputs, {}, result, executable));

  TF_EXPECT_OK(PopulateCtxOutputsFromPjRtExecutableOutputs(
      inputs, {}, *result, execute_outputs, context_.get()));

  Tensor* expected = CreateHostTensor<int32>(TensorShape({1, 3}), {5, 7, 9});
  test::ExpectTensorEqual<int32>(*expected, *GetOutput(0));
}

TEST_F(PjRtExecutionUtilTest, PopulateCtxOutputsVariableInputs) {
  XlaOpRegistry::RegisterCompilationKernels();
  TF_EXPECT_OK(NodeDefBuilder("AddV2", "AddV2")
                   .Input(FakeInput(DT_INT32))
                   .Input(FakeInput(DT_INT32))
                   .Attr("T", DT_INT32)
                   .Device("/job:localhost/replica:0/task:0/device:XLA_CPU:0")
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  AddVariableInput<int32>("var1", TensorShape({1, 2}), {1, 2});
  AddVariableInput<int32>("var2", TensorShape({1, 2}), {3, 4});

  CreateContext();

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({1, 2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({1, 2});

  const XlaCompiler::CompilationResult* result;
  xla::PjRtLoadedExecutable* executable;
  CompileToExecutable(args, &result, &executable);

  std::vector<const Tensor*> inputs = InputsFromContext(context_.get());
  std::vector<int> variables_indices =
      GetResourceVariableIndicesFromContext(context_.get());
  std::vector<VariableInfo> variables;
  variables.reserve(variables_indices.size());
  TF_ASSERT_OK(GetVariableInfosFromInputs(context_->resource_manager(),
                                          context_->device(), inputs,
                                          variables_indices, &variables));
  TF_ASSERT_OK_AND_ASSIGN(auto execute_outputs,
                          RunExecutable(inputs, variables, result, executable));
  TF_EXPECT_OK(PopulateCtxOutputsFromPjRtExecutableOutputs(
      inputs, variables, *result, execute_outputs, context_.get()));

  Tensor* expected = CreateHostTensor<int32>(TensorShape({1, 2}), {4, 6});
  test::ExpectTensorEqual<int32>(*expected, *GetOutput(0));
}

TEST_F(PjRtExecutionUtilTest, PopulateCtxOutputsResourceUpdates) {
  XlaOpRegistry::RegisterCompilationKernels();
  TF_EXPECT_OK(NodeDefBuilder("AssignAddVariableOp", "AssignAddVariableOp")
                   .Input(FakeInput(DT_RESOURCE))
                   .Input(FakeInput(DT_INT32))
                   .Attr("dtype", DT_INT32)
                   .Device("/job:localhost/replica:0/task:0/device:XLA_CPU:0")
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());

  AddVariableInput<int32>("var", TensorShape({1, 3}), {1, 2, 3});
  Tensor* a = CreateDeviceTensor<int32>(TensorShape({1, 3}), {2, 2, 2});
  inputs_.push_back({nullptr, a});

  CreateContext();

  std::vector<const Tensor*> inputs = InputsFromContext(context_.get());
  std::vector<int> variables_indices =
      GetResourceVariableIndicesFromContext(context_.get());
  std::vector<VariableInfo> variables;
  variables.reserve(variables_indices.size());
  TF_ASSERT_OK(GetVariableInfosFromInputs(context_->resource_manager(),
                                          context_->device(), inputs,
                                          variables_indices, &variables));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> constant_input_indices,
                          GetConstantInputIndicesFromContext(context_.get()));
  TF_ASSERT_OK(LockVariables(absl::MakeSpan(variables)));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<XlaCompiler::Argument> args,
      XlaComputationLaunchContext::BuildXlaCompilerArguments(
          constant_input_indices, inputs, variables,
          static_cast<Device*>(context_->device())));

  const XlaCompiler::CompilationResult* result;
  xla::PjRtLoadedExecutable* executable;
  CompileToExecutable(args, &result, &executable);
  TF_ASSERT_OK_AND_ASSIGN(auto execute_outputs,
                          RunExecutable(inputs, variables, result, executable));

  TF_EXPECT_OK(PopulateCtxOutputsFromPjRtExecutableOutputs(
      inputs, variables, *result, execute_outputs, context_.get()));

  // Verify that there are no outputs.
  EXPECT_EQ(context_->num_outputs(), 0);

  // Verify that the original variable was updated.
  ResourceMgr* rm = device_->resource_manager();
  Var* var = nullptr;
  TF_ASSERT_OK(rm->Lookup(rm->default_container(), "var", &var));
  core::ScopedUnref var_ref(var);

  Tensor* device_tensor = var->tensor();
  Tensor* host_tensor = new Tensor(host_allocator_, device_tensor->dtype(),
                                   device_tensor->shape());
  tensors_.push_back(host_tensor);
  TF_ASSERT_OK(device_context_->CopyDeviceTensorToCPUSync(
      device_tensor, "", device_, host_tensor));

  Tensor* expected = CreateHostTensor<int32>(TensorShape({1, 3}), {3, 4, 5});
  test::ExpectTensorEqual<int32>(*expected, *host_tensor);
}

TEST_F(PjRtExecutionUtilTest, GetDeviceOrdinal) {
  EXPECT_EQ(GetDeviceOrdinal(device_), 0);
}

TEST(XlaLaunchUtilTest, GetPjRtExecuteOptions) {
  xla::ExecuteOptions options = GetPjRtExecuteOptions();
  EXPECT_FALSE(options.arguments_are_tupled);
  EXPECT_TRUE(options.untuple_result);
  EXPECT_TRUE(options.use_major_to_minor_data_layout_for_callbacks);
}

}  // namespace
}  // namespace tensorflow
