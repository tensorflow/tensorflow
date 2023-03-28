/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_selector.h"

#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tfrt/cpu/core_runtime/null_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

using ::tensorflow::AbstractTensorHandle;
using ::tensorflow::Allocator;
using ::tensorflow::AllocatorAttributes;
using ::tensorflow::AttrBuilder;
using ::tensorflow::DataType;
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::DeviceAttributes;
using ::tensorflow::DynamicDeviceMgr;
using ::tensorflow::EagerContext;
using ::tensorflow::ImmediateExecutionOperation;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::SessionOptions;
using ::tensorflow::Status;

constexpr char kFullCPU[] = "/job:a/replica:0/task:0/device:CPU:0";
constexpr char kFullGPU[] = "/job:a/replica:0/task:0/device:FakeGPU:0";

////////////////////////////////////////////////////////////////////////////////
//
// Op, kernel to set up the environment.
//
// The Placer uses information about the op (input types),
// kernel (device constraints). To avoid depending on the full runtime, we
// define dummy implementations of these, and register them with the
// runtime.
//
////////////////////////////////////////////////////////////////////////////////

// A dummy OpKernel that is used to register ops on different devices.
class DummyOp : public OpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

// Register the following ops so they can be added to a Graph, and
// kernels so that they can be placed on particular device types.
REGISTER_OP("InvalidOp").Output("o: Ref(float)");

REGISTER_OP("TestOp").Output("o: Ref(float)");
REGISTER_KERNEL_BUILDER(Name("TestOp").Device(DEVICE_CPU).Priority(1), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestOp").Device("FakeGPU").Priority(2), DummyOp);

static tensorflow::Device* CreateDevice(const char* type, const char* name) {
  class FakeDevice : public tensorflow::Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    Status Sync() override { return ::tensorflow::OkStatus(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class FakeTensorHandle : public tensorflow::ImmediateExecutionTensorHandle {
 public:
  explicit FakeTensorHandle(string_view device_name, tensorflow::DataType dtype)
      : ImmediateExecutionTensorHandle(kTfrt),
        device_name_(device_name),
        dtype_(dtype) {}

  void Release() { Unref(); }

  tensorflow::DataType DataType() const override { return dtype_; }
  Status Shape(tensorflow::PartialTensorShape* shape) const override {
    int64_t dim_sizes[] = {1};
    return tensorflow::PartialTensorShape::MakePartialShape(dim_sizes, 1,
                                                            shape);
  }
  Status NumDims(int* num_dims) const override {
    *num_dims = 1;
    return ::tensorflow::OkStatus();
  }
  Status NumElements(int64_t* num_elements) const override {
    *num_elements = 1;
    return ::tensorflow::OkStatus();
  }
  Status Dim(int dim_index, int64_t* dim) const override {
    llvm_unreachable("unimplemented method.");
  }

  const char* DeviceName(Status* status) const override {
    return device_name_.c_str();
  }
  const char* BackingDeviceName(Status* status) const override {
    llvm_unreachable("unimplemented method.");
  }
  const char* DeviceType(Status* status) const override {
    llvm_unreachable("unimplemented method.");
  }
  int DeviceId(Status* status) const override {
    llvm_unreachable("unimplemented method.");
  }
  tensorflow::AbstractTensorInterface* Resolve(Status* status) override {
    llvm_unreachable("unimplemented method.");
  }
  ImmediateExecutionTensorHandle* Copy() {
    Ref();
    return this;
  }
  // Return default (TFT_UNSET) full type information. This could be updated in
  // the future if full type information is needed.
  tensorflow::FullTypeDef FullType() const override {
    return tensorflow::FullTypeDef();
  }

  static bool classof(const AbstractTensorHandle* ptr) { return true; }

 private:
  std::string device_name_;
  tensorflow::DataType dtype_;
};

class FakeOperation : public ImmediateExecutionOperation {
 public:
  explicit FakeOperation() : ImmediateExecutionOperation(kTfrt) {}
  ~FakeOperation() override {}

  void Release() override { delete this; }

  void Clear() override { args_.clear(); }

  tensorflow::ImmediateExecutionContext* GetContext() const override {
    return nullptr;
  }

  bool HasCustomDeviceInput() const override { return false; }

  Status Reset(const char* op, const char* raw_device_name) override {
    op_name_ = op;
    device_name_ = raw_device_name;
    attrs_.Reset(op);
    args_.clear();
    return ::tensorflow::OkStatus();
  }
  const std::string& Name() const override { return op_name_; }
  const std::string& DeviceName() const override { return device_name_; }
  tensorflow::Status SetDeviceName(const char* name) override {
    device_name_ = name;
    return ::tensorflow::OkStatus();
  }

  Status AddInput(AbstractTensorHandle* input) override {
    input->Ref();
    args_.push_back(tensorflow::core::RefCountPtr<FakeTensorHandle>(
        static_cast<FakeTensorHandle*>(input)));
    attrs_.NumInputs(args_.size());
    return ::tensorflow::OkStatus();
  }
  Status SetInput(size_t index,
                  tensorflow::ImmediateExecutionTensorHandle* input) override {
    llvm_unreachable("unimplemented method.");
  }
  Status AddInputList(absl::Span<AbstractTensorHandle* const> inputs) override {
    llvm_unreachable("unimplemented method.");
  }
  absl::Span<tensorflow::ImmediateExecutionTensorHandle* const> GetInputs()
      const override {
    return absl::MakeSpan(
        reinterpret_cast<tensorflow::ImmediateExecutionTensorHandle* const*>(
            args_.data()),
        args_.size());
  }
  Status Execute(absl::Span<AbstractTensorHandle*> retvals,
                 int* num_retvals) override {
    llvm_unreachable("unimplemented method.");
  }
  const tensorflow::OpDef* OpDef() const override {
    llvm_unreachable("unimplemented method.");
  }
  const tensorflow::AbstractOpAttrs* GetOpAttrs() const override {
    llvm_unreachable("unimplemented method.");
  }
  void AddAttrs(const tensorflow::AbstractOpAttrs* op_attrs) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrString(const char* attr_name, const char* data,
                       size_t length) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrInt(const char* attr_name, int64_t value) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFloat(const char* attr_name, float value) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrBool(const char* attr_name, bool value) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrType(const char* attr_name,
                     tensorflow::DataType value) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrShape(const char* attr_name, const int64_t* dims,
                      const int num_dims) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFunction(const char* attr_name,
                         const AbstractOperation* value) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFunctionName(const char* attr_name, const char* data,
                             size_t length) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrTensor(const char* attr_name,
                       tensorflow::AbstractTensorInterface* tensor) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrStringList(const char* attr_name, const void* const* values,
                           const size_t* lengths, int num_values) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFloatList(const char* attr_name, const float* values,
                          int num_values) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrIntList(const char* attr_name, const int64_t* values,
                        int num_values) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrTypeList(const char* attr_name,
                         const tensorflow::DataType* values,
                         int num_values) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrBoolList(const char* attr_name, const unsigned char* values,
                         int num_values) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrShapeList(const char* attr_name, const int64_t** dims,
                          const int* num_dims, int num_values) override {
    llvm_unreachable("unimplemented method.");
  }
  Status SetAttrFunctionList(
      const char* attr_name,
      absl::Span<const AbstractOperation*> values) override {
    llvm_unreachable("unimplemented method.");
  }

  Status InputLength(const char* input_name, int* length) override {
    llvm_unreachable("unimplemented method.");
  }
  Status OutputLength(const char* output_name, int* length) override {
    llvm_unreachable("unimplemented method.");
  }

  void SetCancellationManager(
      tensorflow::CancellationManager* cancellation_manager) override {
    llvm_unreachable("unimplemented method.");
  }

  void SetStackTrace(tensorflow::ManagedStackTrace stack_trace) override {
    llvm_unreachable("unimplemented method.");
  }

  absl::optional<tensorflow::ManagedStackTrace> GetStackTrace() override {
    llvm_unreachable("unimplemented method.");
  }

  void SetStepId(int64_t step_id) override {
    llvm_unreachable("unimplemented method.");
  }

  static bool classof(const AbstractOperation* ptr) { return true; }

  AttrBuilder* GetAttrs() { return &attrs_; }

 private:
  std::string op_name_;
  std::string device_name_;
  llvm::SmallVector<tensorflow::core::RefCountPtr<FakeTensorHandle>, 8> args_;
  AttrBuilder attrs_;
};

static std::unique_ptr<CoreRuntime> CreateCoreRuntime() {
  auto diag_handler = [](const DecodedDiagnostic& diag) {
    LOG(ERROR) << "Encountered runtime error: " << diag.message() << "\n";
  };
  auto corert =
      CoreRuntime::Create(diag_handler, tfrt::CreateMallocAllocator(),
                          tfrt::CreateMultiThreadedWorkQueue(
                              /*num_threads=*/4, /*num_blocking_threads=*/64),
                          kFullCPU);

  assert(corert);
  return std::move(*corert);
}

class SelectorTest : public ::testing::Test {
 public:
  SelectorTest() {
    device_manager_ = new DynamicDeviceMgr();
    std::vector<std::unique_ptr<tensorflow::Device>> added_devices;
    SessionOptions opts;

    // Have to use real CPU device. Other, ctx->HostCPU() will return invalid
    // device.
    added_devices.emplace_back(CreateDevice(tensorflow::DEVICE_CPU, kFullCPU));
    added_devices.emplace_back(CreateDevice("FakeGPU", kFullGPU));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));

    SessionOptions options;
    options.config.set_log_device_placement(true);
    options.config.set_allow_soft_placement(true);
    eager_context_ = new EagerContext(
        options,
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /* async */ false, device_manager_,
        /* device_mgr_owned */ false, /* rendezvous */ nullptr,
        /* cluster_flr */ nullptr, /*collective_executor_mgr=*/nullptr,
        /*run_eager_op_as_function=*/true);
    corert_ = CreateCoreRuntime();
    fallback_op_handler_ = CreateOpHandler();
    cpu_op_handler_ = CreateOpHandler();
    gpu_op_handler_ = CreateOpHandler();
    corert_->RegisterOpHandler(kFullCPU, cpu_op_handler_);
    corert_->RegisterOpHandler(kFullGPU, gpu_op_handler_);

    selector_ = std::make_unique<EagerOpHandlerSelector>(
        corert_.get(), eager_context_, fallback_op_handler_,
        /*pin_small_ops_to_cpu=*/true);
  }

  ~SelectorTest() override {
    delete device_manager_;
    if (eager_context_) {
      eager_context_->Unref();
    }
  }

  EagerOpHandlerSelector* selector() { return selector_.get(); }

  void Init() {}

 protected:
  OpHandler* CreateOpHandler() {
    auto expected_op_handler = tfrt::CreateNullOpHandler(corert_.get());
    assert(expected_op_handler);
    return std::move(expected_op_handler.get());
  }

  DynamicDeviceMgr* device_manager_;
  EagerContext* eager_context_;
  std::unique_ptr<CoreRuntime> corert_;
  OpHandler* fallback_op_handler_;
  OpHandler* cpu_op_handler_;
  OpHandler* gpu_op_handler_;
  std::unique_ptr<EagerOpHandlerSelector> selector_;
};

TEST_F(SelectorTest, PinSmallOpToCpuTest) {
  auto op = std::make_unique<FakeOperation>();
  tensorflow::core::RefCountPtr<FakeTensorHandle> cpu_tensor(
      new FakeTensorHandle(kFullCPU, tensorflow::DT_INT32));
  tensorflow::core::RefCountPtr<FakeTensorHandle> gpu_tensor(
      new FakeTensorHandle(kFullGPU, tensorflow::DT_INT32));

  tensorflow::Status s;
  TF_ASSERT_OK(op->Reset("TestOp", kFullGPU));
  TF_ASSERT_OK(op->AddInput(cpu_tensor.get()));
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, ::tensorflow::OkStatus());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, cpu_op_handler_);

  op_handler = nullptr;
  TF_ASSERT_OK(op->Reset("TestOp", kFullGPU));
  TF_ASSERT_OK(op->AddInput(gpu_tensor.get()));
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, ::tensorflow::OkStatus());
  ASSERT_FALSE(static_cast<bool>(op_handler));
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s, ::tensorflow::OkStatus());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

TEST_F(SelectorTest, PinResourceTest) {
  auto op = std::make_unique<FakeOperation>();
  tensorflow::core::RefCountPtr<FakeTensorHandle> cpu_tensor(
      new FakeTensorHandle(kFullCPU, tensorflow::DT_RESOURCE));
  tensorflow::core::RefCountPtr<FakeTensorHandle> gpu_tensor(
      new FakeTensorHandle(kFullGPU, tensorflow::DT_RESOURCE));

  tensorflow::Status s;
  TF_ASSERT_OK(op->Reset("TestOp", kFullGPU));
  TF_ASSERT_OK(op->AddInput(cpu_tensor.get()));
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, ::tensorflow::OkStatus());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, cpu_op_handler_);

  op_handler = nullptr;
  TF_ASSERT_OK(op->Reset("TestOp", kFullCPU));
  TF_ASSERT_OK(op->AddInput(gpu_tensor.get()));
  s = selector()->SelectFromArguments(*op, &op_handler);
  ASSERT_EQ(s, ::tensorflow::OkStatus());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

TEST_F(SelectorTest, InvalidDeviceNameTest) {
  auto op = std::make_unique<FakeOperation>();

  TF_ASSERT_OK(op->Reset("TestOp", "invalid_device_name"));

  tensorflow::Status s;
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_FALSE(static_cast<bool>(op_handler));
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "Failed to parse device name"));
}

TEST_F(SelectorTest, SoftPlacementTest) {
  auto op = std::make_unique<FakeOperation>();

  TF_ASSERT_OK(op->Reset("TestOp", "/device:FakeGPU:99"));
  tensorflow::Status s;
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s, ::tensorflow::OkStatus());
  ASSERT_TRUE(static_cast<bool>(op_handler)) << StrCat(s.error_message());
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

TEST_F(SelectorTest, HigherPriorityDeviceTest) {
  auto op = std::make_unique<FakeOperation>();

  tensorflow::Status s;
  TF_ASSERT_OK(op->Reset("TestOp", ""));
  OpHandler* op_handler = nullptr;
  s = selector()->SelectFromNodeDef(*op, &op->GetAttrs()->BuildNodeDef(),
                                    &op_handler);
  ASSERT_EQ(s, ::tensorflow::OkStatus());
  ASSERT_TRUE(static_cast<bool>(op_handler));
  ASSERT_EQ(op_handler, gpu_op_handler_);
}

}  // namespace
}  // namespace tf
}  // namespace tfrt
