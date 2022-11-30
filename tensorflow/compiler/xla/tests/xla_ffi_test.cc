/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/runtime/ffi/ffi_api.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {

using ::xla::runtime::ffi::Ffi;
using ::xla::runtime::ffi::FfiStatus;
using ::xla::runtime::ffi::StridedBufferArg;

class FfiTest : public HloTestBase {
 protected:
  Shape tensor_4xf32_ = ShapeUtil::MakeShape(F32, {4});
};

// TODO(ezhulenev): Add support for stateless XLA FFI modules.
struct EmptyState {};

// XLA FFI module encapsulating FFI functions exported to the runtime.
struct TestModule : public runtime::ffi::StatefulModule<EmptyState> {
  using Base = runtime::ffi::StatefulModule<EmptyState>;

  explicit TestModule(const XLA_FFI_Api* api)
      : Base(api, "xla-ffi-module", {{"test.ffi", FFI_Impl}}) {}

  std::unique_ptr<EmptyState> CreateState() const final {
    return std::make_unique<EmptyState>();
  }

  // XLA runtime binding for the C++ function.
  XLA_FFI_DEFINE_FUNCTION(FFI_Impl, Impl,
                          Ffi::Bind("test.ffi")
                              .Arg<StridedBufferArg>()
                              .Arg<StridedBufferArg>()
                              .Arg<StridedBufferArg>()
                              .Attr<float>("foo"));

  // Typed XLA FFI function handler that will be registered with the runtime.
  //
  // WARNING: Buffer arguments are placed on the GPU device and we can't touch
  // the memory they are pointing to on the host.
  static FfiStatus Impl(StridedBufferArg input0, StridedBufferArg input1,
                        StridedBufferArg out, float foo);
};

// Observe FFI arguments by adding them to the static vector.
static std::vector<std::string>* GetFfiArgs() {
  static auto* args = new std::vector<std::string>();
  return args;
}

FfiStatus TestModule::Impl(StridedBufferArg input0, StridedBufferArg input1,
                           StridedBufferArg out, float foo) {
  auto* args = GetFfiArgs();
  args->push_back(std::to_string(foo));
  args->push_back(input0.ToString());
  args->push_back(input1.ToString());
  args->push_back(out.ToString());
  return FfiStatus::Ok();
}

// TODO(ezhulenev): We have to register stubs in the XLA custom call registry to
// suppress errors during Thunk emissions. This stub should never be called,
// and instead XLA will call into registered FFI handlers. Remove this hack!
static void Abort() { LOG(FATAL) << "Custom call stub must never be called"; }
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("test.ffi", Abort, "CUDA");

XLA_TEST_F(FfiTest, Basic) {
  // Register XLA FFI module with the runtime.
  TestModule ffi_module(GetXlaFfiApi());

  absl::string_view mlir_module_str = R"(
  module @xla_ffi {
    func.func public @main(%arg0: tensor<4xf32>,
                           %arg1: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>

      %1 = "mhlo.custom_call"(%arg0, %arg1)
        { api_version = 4 : i32, call_target_name = "test.ffi",
          has_side_effect = true, backend_config = {foo = 42.0 : f32}
        } : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>)

      return %0 : tensor<4xf32>
    }
  }
  )";

  // Convert Mhlo to Hlo Module.
  XlaComputation xla_computation;
  TF_ASSERT_OK(ParseMlirModuleStringAndConvertToXlaComputation(
      mlir_module_str, xla_computation, false, false));

  HloModuleProto proto = xla_computation.proto();

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_xla_runtime_executable(true);

  // Instantiate HloModule from the computation.
  auto config = HloModule::CreateModuleConfigFromProto(proto, debug_options);
  TF_ASSERT_OK(config.status());

  auto module = HloModule::CreateFromProto(proto, *config);
  TF_ASSERT_OK(module.status());

  // Prepare test inputs.
  Array<float> arr0({1.0f, 2.0f, 3.0f, 4.0f});
  Array<float> arr1({4.0f, 3.0f, 2.0f, 1.0f});

  Literal arg0 = LiteralUtil::CreateFromArray(arr0);
  Literal arg1 = LiteralUtil::CreateFromArray(arr1);

  Literal result = ExecuteAndTransfer(std::move(*module), {&arg0, &arg1});

  // Check that `mhlo.add` was executed.
  LiteralTestUtil::ExpectR1Equal<float>({5.0f, 5.0f, 5.0f, 5.0f}, result);

  // Check that FFI handler was also executed.
  auto* args = GetFfiArgs();
  ASSERT_EQ(args->size(), 4);
  ASSERT_TRUE(absl::StartsWith(args->at(0), "42"));
  ASSERT_EQ(args->at(1), "Buffer: dtype=f32 sizes=[4] strides=[1]");
  ASSERT_EQ(args->at(2), "Buffer: dtype=f32 sizes=[4] strides=[1]");
  ASSERT_EQ(args->at(3), "Buffer: dtype=f32 sizes=[4] strides=[1]");
}

}  // namespace gpu
}  // namespace xla
