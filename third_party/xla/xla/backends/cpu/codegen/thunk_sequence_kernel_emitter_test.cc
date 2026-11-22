/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/thunk_sequence_kernel_emitter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/conditional_thunk.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/runtime/kernel_thunk.h"
#include "xla/backends/cpu/runtime/rng_state_resource.h"
#include "xla/backends/cpu/runtime/rng_state_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/while_thunk.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/debug_options_flags.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/decision.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// Kernels written against the XLA:CPU Kernel C API (see kernel_c_api.h).
constexpr absl::string_view kKernelTypes = R"(
  %Dim3 = type { i64, i64, i64 }
  %Arg = type { ptr, i64 }
  %Frame = type { ptr, ptr, i64, ptr }
)";

// args[1][0] += args[0][0], as i32.
constexpr absl::string_view kAdd = R"(
  define ptr @add(ptr %frame) {
    %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
    %args = load ptr, ptr %args_gep
    %in_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
    %in = load ptr, ptr %in_gep
    %out_gep = getelementptr %Arg, ptr %args, i32 1, i32 0
    %out = load ptr, ptr %out_gep
    %a = load i32, ptr %in
    %b = load i32, ptr %out
    %c = add i32 %a, %b
    store i32 %c, ptr %out
    ret ptr null
  }
)";

// Appends x + 10 * y + 100 * z of the workgroup id to the list in args[0],
// whose first element is its length.
constexpr absl::string_view kRecord = R"(
  define ptr @record(ptr %frame) {
    %id_gep = getelementptr %Frame, ptr %frame, i32 0, i32 1
    %id = load ptr, ptr %id_gep
    %x_gep = getelementptr %Dim3, ptr %id, i32 0, i32 0
    %y_gep = getelementptr %Dim3, ptr %id, i32 0, i32 1
    %z_gep = getelementptr %Dim3, ptr %id, i32 0, i32 2
    %x = load i64, ptr %x_gep
    %y = load i64, ptr %y_gep
    %z = load i64, ptr %z_gep
    %y10 = mul i64 %y, 10
    %z100 = mul i64 %z, 100
    %xy = add i64 %x, %y10
    %xyz = add i64 %xy, %z100
    %v = trunc i64 %xyz to i32
    %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
    %args = load ptr, ptr %args_gep
    %out_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
    %out = load ptr, ptr %out_gep
    %n = load i32, ptr %out
    %n1 = add i32 %n, 1
    %slot = getelementptr i32, ptr %out, i32 %n1
    store i32 %v, ptr %slot
    store i32 %n1, ptr %out
    ret ptr null
  }
)";

// args[1] = args[0] < 5, as i32 to pred.
constexpr absl::string_view kLessThanFive = R"(
  define ptr @lt5(ptr %frame) {
    %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
    %args = load ptr, ptr %args_gep
    %in_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
    %in = load ptr, ptr %in_gep
    %out_gep = getelementptr %Arg, ptr %args, i32 1, i32 0
    %out = load ptr, ptr %out_gep
    %v = load i32, ptr %in
    %lt = icmp slt i32 %v, 5
    %p = zext i1 %lt to i8
    store i8 %p, ptr %out
    ret ptr null
  }
)";

// Writes 7 to args[0] and returns an error.
constexpr absl::string_view kFail = R"(
  define ptr @fail(ptr %frame) {
    %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
    %args = load ptr, ptr %args_gep
    %out_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
    %out = load ptr, ptr %out_gep
    store i32 7, ptr %out
    ret ptr inttoptr (i64 1 to ptr)
  }
)";

// Two kernels that each define a non kernel function @helper, returning a
// different constant. They store it to args[0][0] and args[0][1].
constexpr absl::string_view kHelperOne = R"(
  define i32 @helper() { ret i32 1 }
  define ptr @one(ptr %frame) {
    %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
    %args = load ptr, ptr %args_gep
    %out_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
    %out = load ptr, ptr %out_gep
    %v = call i32 @helper()
    store i32 %v, ptr %out
    ret ptr null
  }
)";
constexpr absl::string_view kHelperTwo = R"(
  define i32 @helper() { ret i32 2 }
  define ptr @two(ptr %frame) {
    %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
    %args = load ptr, ptr %args_gep
    %out_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
    %out = load ptr, ptr %out_gep
    %second = getelementptr i32, ptr %out, i32 1
    %v = call i32 @helper()
    store i32 %v, ptr %second
    ret ptr null
  }
)";

class ThunkSequenceKernelEmitterTest : public ::testing::Test {
 protected:
  ThunkSequenceKernelEmitterTest()
      : allocation_(/*index=*/0, /*size=*/kMemorySize, /*color=*/0) {}

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        target_machine_,
        IrCompiler::InferTargetMachine(
            llvm::TargetOptions(), llvm::CodeGenOptLevel::Default,
            TargetMachineOptions(GetDebugOptionsFromFlags())));
    features_ = std::make_unique<TargetMachineFeatures>(target_machine_.get());
  }

  // An s32[n] buffer at byte `offset`.
  ShapedSlice Int(int64_t offset, int64_t n = 1) {
    return {BufferAllocation::Slice(&allocation_, offset, 4 * n),
            ShapeUtil::MakeShape(S32, {n})};
  }

  ShapedSlice Pred(int64_t offset) {
    return {BufferAllocation::Slice(&allocation_, offset, 1),
            ShapeUtil::MakeShape(PRED, {})};
  }

  static absl::StatusOr<std::unique_ptr<Thunk>> Kernel(
      absl::string_view name, std::vector<ShapedSlice> arguments,
      std::vector<ShapedSlice> results, NumWorkGroups num_workgroups = {}) {
    return KernelThunk::Create({std::string(name)}, arguments, results, name,
                               num_workgroups, /*invariant_arguments=*/{});
  }

  static absl::StatusOr<ThunkSequence> SingleKernel(
      absl::string_view name, std::vector<ShapedSlice> arguments,
      std::vector<ShapedSlice> results, NumWorkGroups num_workgroups = {}) {
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<Thunk> thunk,
        Kernel(name, std::move(arguments), std::move(results), num_workgroups));
    ThunkSequence sequence;
    sequence.push_back(std::move(thunk));
    return sequence;
  }

  static ThunkSequence Sequence(std::vector<std::unique_ptr<Thunk>> thunks) {
    ThunkSequence sequence;
    for (auto& thunk : thunks) {
      sequence.push_back(std::move(thunk));
    }
    return sequence;
  }

  static absl::StatusOr<LlvmKernelSource> Parse(absl::string_view ir) {
    auto context = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic diagnostic;
    std::string text = absl::StrCat(kKernelTypes, ir);
    std::unique_ptr<llvm::Module> module = llvm::parseAssembly(
        llvm::MemoryBufferRef(text, "kernel"), diagnostic, *context);
    if (module == nullptr) {
      return absl::InternalError(diagnostic.getMessage().str());
    }
    SetModuleMemoryRegionName(*module, "test");
    return LlvmKernelSource(std::move(context), std::move(module));
  }

  // The kernel modules in `ir`, each named by its last defined function.
  static std::vector<ThunkSequenceKernelEmitter::KernelModule> KernelModules(
      absl::Span<const LlvmKernelSource> sources) {
    std::vector<ThunkSequenceKernelEmitter::KernelModule> kernels;
    for (const LlvmKernelSource& source : sources) {
      const llvm::Module* module = source.module();
      kernels.push_back(
          {module->getFunctionList().back().getName().str(), module});
    }
    return kernels;
  }

  // The distinct non empty slices used by `thunks`, which become kernel
  // results.
  static KernelSpec::Buffers Parameters(const ThunkSequence& thunks) {
    KernelSpec::Buffers parameters;
    absl::flat_hash_set<BufferAllocation::Slice> seen;
    for (const BufferUse& use : thunks.buffer_uses()) {
      if (!use.slice().empty() && seen.insert(use.slice()).second) {
        parameters.push_back({use.slice(), use.shape()});
      }
    }
    return parameters;
  }

  absl::StatusOr<LlvmKernelSource> Emit(const ThunkSequence& thunks,
                                        const KernelSpec::Buffers& parameters) {
    return Emit(thunks, /*arguments=*/{}, parameters);
  }

  absl::StatusOr<LlvmKernelSource> Emit(const ThunkSequence& thunks,
                                        const KernelSpec::Buffers& arguments,
                                        const KernelSpec::Buffers& results) {
    ThunkSequenceKernelEmitter emitter(
        "main", "test", arguments, results, &thunks, features_.get(),
        {/*prefer_vector_width=*/256,
         /*generate_unique_c_style_kernel_entry_points=*/false});
    ABSL_ASSIGN_OR_RETURN(KernelDefinition<LlvmKernelSource> definition,
                     emitter.EmitKernelDefinition());
    return std::move(definition).TakeSource();
  }

  // Compiles `thunks` with the kernels defined by `ir` and runs the result on
  // `memory_`. Returns whether the kernel returned an error.
  absl::StatusOr<bool> Run(const ThunkSequence& thunks,
                           absl::Span<const absl::string_view> ir) {
    return Run(thunks, ir, Parameters(thunks));
  }

  ABSL_ATTRIBUTE_NO_SANITIZE_CFI absl::StatusOr<bool> Run(
      const ThunkSequence& thunks, absl::Span<const absl::string_view> ir,
      const KernelSpec::Buffers& parameters) {
    std::vector<LlvmKernelSource> sources;
    absl::flat_hash_set<std::string> names;
    for (absl::string_view kernel : ir) {
      ABSL_ASSIGN_OR_RETURN(LlvmKernelSource parsed, Parse(kernel));
      sources.push_back(std::move(parsed));
    }
    std::vector<ThunkSequenceKernelEmitter::KernelModule> kernels =
        KernelModules(sources);
    for (const auto& kernel : kernels) {
      names.insert(kernel.name);
    }

    if (Decision d =
            ThunkSequenceKernelEmitter::CanEmit(thunks, names, {}, parameters);
        d.IsForbidden()) {
      return absl::InvalidArgumentError(d.Explain());
    }
    ABSL_ASSIGN_OR_RETURN(LlvmKernelSource source, Emit(thunks, parameters));
    ABSL_RETURN_IF_ERROR(
        ThunkSequenceKernelEmitter::LinkKernels(*source.module(), kernels));

    std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
        llvm::TargetOptions(),
        IrCompiler::Options{llvm::CodeGenOptLevel::Default,
                            /*optimize_for_size=*/false,
                            TargetMachineOptions(GetDebugOptionsFromFlags())},
        IrCompiler::CompilationHooks());
    ABSL_ASSIGN_OR_RETURN(JitCompiler compiler,
                     JitCompiler::Create({}, std::move(ir_compiler)));
    ABSL_RETURN_IF_ERROR(compiler.AddModule(std::move(source).thread_safe_module()));
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<FunctionLibrary> library,
        std::move(compiler).Compile(
            {FunctionLibrary::Sym<FunctionLibrary::Kernel>("main")}));
    ABSL_ASSIGN_OR_RETURN(FunctionLibrary::Kernel * kernel,
                     library->ResolveFunction<FunctionLibrary::Kernel>("main"));

    std::vector<XLA_CPU_KernelArg> args;
    for (const ShapedSlice& parameter : parameters) {
      args.push_back({Bytes() + parameter.slice.offset(),
                      static_cast<size_t>(parameter.slice.size())});
    }
    XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
    XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
    XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id,
                                     args.size(), args.data()};
    return (*kernel)(&frame) != nullptr;
  }

  uint8_t* Bytes() {
    return static_cast<uint8_t*>(static_cast<void*>(memory_.data()));
  }

  int32_t& Value(const ShapedSlice& slice, int64_t index = 0) {
    return memory_[slice.slice.offset() / sizeof(int32_t) + index];
  }

  static constexpr int64_t kMemorySize = 256;

  BufferAllocation allocation_;
  std::unique_ptr<llvm::TargetMachine> target_machine_;
  std::unique_ptr<TargetMachineFeatures> features_;
  alignas(64) std::array<int32_t, kMemorySize / sizeof(int32_t)> memory_ = {};
};

TEST_F(ThunkSequenceKernelEmitterTest, KernelsRunInOrder) {
  ShapedSlice a = Int(0), b = Int(64), c = Int(128);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t0, Kernel("add", {a}, {b}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t1, Kernel("add", {b}, {c}));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(t0));
  thunks.push_back(std::move(t1));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Value(a) = 2;
  Value(b) = 3;
  Value(c) = 4;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(b), 5);
  EXPECT_EQ(Value(c), 9);
}

TEST_F(ThunkSequenceKernelEmitterTest, WorkgroupsRunInLaunchOrder) {
  ShapedSlice list = Int(0, 9);
  ASSERT_OK_AND_ASSIGN(
      ThunkSequence sequence,
      SingleKernel("record", {}, {list}, NumWorkGroups{2, 2, 2}));

  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kRecord}));
  EXPECT_FALSE(error);
  std::vector<int32_t> ids(&Value(list, 1), &Value(list, 9));
  EXPECT_EQ(Value(list), 8);
  EXPECT_THAT(ids, ElementsAre(0, 1, 10, 11, 100, 101, 110, 111));
}

TEST_F(ThunkSequenceKernelEmitterTest, WhileEvaluatesCondition) {
  ShapedSlice counter = Int(0), one = Int(64);
  ShapedSlice pred = Pred(128);
  ASSERT_OK_AND_ASSIGN(ThunkSequence cond,
                       SingleKernel("lt5", {counter}, {pred}));
  ASSERT_OK_AND_ASSIGN(ThunkSequence body,
                       SingleKernel("add", {one}, {counter}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<WhileThunk> while_thunk,
                       WhileThunk::Create({"while"}, pred.slice,
                                          std::move(cond), std::move(body)));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(while_thunk));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Value(counter) = 0;
  Value(one) = 1;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd, kLessThanFive}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(counter), 5);
}

// As WhileThunk, a known trip count runs the body without the condition, which
// here would fail.
TEST_F(ThunkSequenceKernelEmitterTest, WhileWithTripCountSkipsCondition) {
  ShapedSlice counter = Int(0), one = Int(64), scratch = Int(192);
  ShapedSlice pred = Pred(128);
  ASSERT_OK_AND_ASSIGN(ThunkSequence cond, SingleKernel("fail", {}, {scratch}));
  ASSERT_OK_AND_ASSIGN(ThunkSequence body,
                       SingleKernel("add", {one}, {counter}));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<WhileThunk> while_thunk,
      WhileThunk::Create({"while"}, pred.slice, std::move(cond),
                         std::move(body), /*trip_count=*/3));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(while_thunk));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Value(counter) = 0;
  Value(one) = 1;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd, kFail}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(counter), 3);
  EXPECT_EQ(Value(scratch), 0);
}

TEST_F(ThunkSequenceKernelEmitterTest, ConditionalSelectsBranch) {
  ShapedSlice index = Int(0), one = Int(64), result = Int(128);
  // Branch i adds 1 to result i + 1 times.
  std::vector<ThunkSequence> branches;
  for (int i = 0; i < 3; ++i) {
    std::vector<std::unique_ptr<Thunk>> thunks;
    for (int j = 0; j <= i; ++j) {
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> thunk,
                           Kernel("add", {one}, {result}));
      thunks.push_back(std::move(thunk));
    }
    branches.push_back(Sequence(std::move(thunks)));
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConditionalThunk> conditional,
                       ConditionalThunk::Create({"conditional"}, index.slice,
                                                std::move(branches)));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(conditional));
  ThunkSequence sequence = Sequence(std::move(thunks));

  // Out of range indices select the last branch.
  for (auto [branch_index, expected] : std::vector<std::pair<int32_t, int32_t>>{
           {0, 1}, {1, 2}, {2, 3}, {-1, 3}, {7, 3}}) {
    SCOPED_TRACE(branch_index);
    Value(index) = branch_index;
    Value(one) = 1;
    Value(result) = 0;
    ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd}));
    EXPECT_FALSE(error);
    EXPECT_EQ(Value(result), expected);
  }
}

TEST_F(ThunkSequenceKernelEmitterTest, ConditionalOnPred) {
  ShapedSlice pred = Pred(0), one = Int(64), result = Int(128), two = Int(192);
  std::vector<ThunkSequence> branches;
  ASSERT_OK_AND_ASSIGN(ThunkSequence on_true,
                       SingleKernel("add", {one}, {result}));
  ASSERT_OK_AND_ASSIGN(ThunkSequence on_false,
                       SingleKernel("add", {two}, {result}));
  branches.push_back(std::move(on_true));
  branches.push_back(std::move(on_false));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConditionalThunk> conditional,
                       ConditionalThunk::Create({"conditional"}, pred.slice,
                                                std::move(branches)));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(conditional));
  ThunkSequence sequence = Sequence(std::move(thunks));

  for (auto [value, expected] :
       std::vector<std::pair<uint8_t, int32_t>>{{1, 1}, {0, 2}}) {
    Bytes()[0] = value;
    Value(one) = 1;
    Value(two) = 2;
    Value(result) = 0;
    ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd}));
    EXPECT_FALSE(error);
    EXPECT_EQ(Value(result), expected);
  }
}

TEST_F(ThunkSequenceKernelEmitterTest, Copy) {
  ShapedSlice src = Int(0, 4), dst = Int(64, 4);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CopyThunk> copy,
      CopyThunk::Create({"copy"}, src.slice, src.shape, dst.slice, dst.shape));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(copy));
  ThunkSequence sequence = Sequence(std::move(thunks));

  for (int i = 0; i < 4; ++i) {
    Value(src, i) = i + 1;
  }
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {}));
  EXPECT_FALSE(error);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(Value(dst, i), i + 1);
  }
}

TEST_F(ThunkSequenceKernelEmitterTest, SelfCopyOmitsMemCpy) {
  ShapedSlice buf = Int(0, 4);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CopyThunk> copy,
      CopyThunk::Create({"copy"}, buf.slice, buf.shape, buf.slice, buf.shape));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(copy));
  ThunkSequence sequence = Sequence(std::move(thunks));

  ASSERT_OK_AND_ASSIGN(LlvmKernelSource source,
                       Emit(sequence, Parameters(sequence)));
  for (const llvm::Function& fn : *source.module()) {
    EXPECT_FALSE(fn.getName().starts_with("llvm.memcpy"));
  }
}

TEST_F(ThunkSequenceKernelEmitterTest, ErrorStopsExecution) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t0, Kernel("fail", {}, {a}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t1, Kernel("add", {a}, {b}));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(t0));
  thunks.push_back(std::move(t1));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Value(b) = 1;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kFail, kAdd}));
  EXPECT_TRUE(error);
  EXPECT_EQ(Value(a), 7);
  EXPECT_EQ(Value(b), 1);
}

// Non kernel functions of different kernels must not clash when linked.
TEST_F(ThunkSequenceKernelEmitterTest, PrivateSymbolsDoNotCollide) {
  ShapedSlice out = Int(0, 2);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t0, Kernel("one", {}, {out}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t1, Kernel("two", {}, {out}));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(t0));
  thunks.push_back(std::move(t1));
  ThunkSequence sequence = Sequence(std::move(thunks));

  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kHelperOne, kHelperTwo}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(out, 0), 1);
  EXPECT_EQ(Value(out, 1), 2);
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsUnknownKernel) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("add", {a}, {b}));
  Decision decision = ThunkSequenceKernelEmitter::CanEmit(
      sequence, /*kernels=*/{}, {}, Parameters(sequence));
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(), HasSubstr("No source for kernel add"));
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsMissingParameter) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("add", {a}, {b}));
  Decision decision =
      ThunkSequenceKernelEmitter::CanEmit(sequence, {"add"}, {a}, {});
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(),
              HasSubstr("is not inside a kernel parameter"));
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsOverlappingParameters) {
  ShapedSlice a = Int(0), b = Int(4);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("add", {a}, {b}));

  Decision partial = ThunkSequenceKernelEmitter::CanEmit(
      sequence, {"add"}, {Int(0, 4)}, {Int(4, 1)});
  EXPECT_TRUE(partial.IsForbidden());
  EXPECT_THAT(partial.Explain(), HasSubstr("overlap"));

  Decision identical = ThunkSequenceKernelEmitter::CanEmit(
      sequence, {"add"}, {Int(0, 4)}, {Int(0, 4)});
  EXPECT_TRUE(identical.IsForbidden());
  EXPECT_THAT(identical.Explain(), HasSubstr("overlap"));

  Decision adjacent =
      ThunkSequenceKernelEmitter::CanEmit(sequence, {"add"}, {a}, {b});
  EXPECT_TRUE(adjacent.IsAllowed());
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsWriteToArgument) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("add", {a}, {b}));
  Decision decision =
      ThunkSequenceKernelEmitter::CanEmit(sequence, {"add"}, {a, b}, {});
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(),
              HasSubstr("is written but lies in argument parameter"));
}

TEST_F(ThunkSequenceKernelEmitterTest, SlicesInsideOneRegion) {
  ShapedSlice region = Int(0, 16);
  ShapedSlice a = Int(16), b = Int(32), c = Int(48);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> add, Kernel("add", {a}, {b}));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CopyThunk> copy,
      CopyThunk::Create({"copy"}, b.slice, b.shape, c.slice, c.shape));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(add));
  thunks.push_back(std::move(copy));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Value(a) = 10;
  Value(b) = 20;
  Value(c) = 0;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd}, {region}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(b), 30);
  EXPECT_EQ(Value(c), 30);
}

TEST_F(ThunkSequenceKernelEmitterTest, WhilePredicateInsideRegion) {
  ShapedSlice counter = Int(0), one = Int(64);
  ShapedSlice pred = Pred(129);
  ShapedSlice pred_region = Int(128, 4);
  ASSERT_OK_AND_ASSIGN(ThunkSequence cond,
                       SingleKernel("lt5", {counter}, {pred}));
  ASSERT_OK_AND_ASSIGN(ThunkSequence body,
                       SingleKernel("add", {one}, {counter}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<WhileThunk> while_thunk,
                       WhileThunk::Create({"while"}, pred.slice,
                                          std::move(cond), std::move(body)));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(while_thunk));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Value(counter) = 0;
  Value(one) = 1;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd, kLessThanFive},
                                       {counter, one, pred_region}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(counter), 5);
}

TEST_F(ThunkSequenceKernelEmitterTest, BodyHasNoaliasParameters) {
  BufferAllocation param(/*index=*/1, /*size=*/64, /*color=*/0);
  param.set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  ShapedSlice a = {BufferAllocation::Slice(&param, 0, 4),
                   ShapeUtil::MakeShape(S32, {})};
  ShapedSlice b = Int(0);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("add", {a}, {b}));

  ASSERT_OK_AND_ASSIGN(LlvmKernelSource source, Emit(sequence, {a}, {b}));
  llvm::Function* body = source.module()->getFunction("main_body");
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->arg_size(), 2);

  EXPECT_TRUE(body->hasParamAttribute(0, llvm::Attribute::NoAlias));
  EXPECT_TRUE(body->hasParamAttribute(0, llvm::Attribute::ReadOnly));
  EXPECT_TRUE(body->hasParamAttribute(1, llvm::Attribute::NoAlias));
  EXPECT_FALSE(body->hasParamAttribute(1, llvm::Attribute::ReadOnly));
  // LLVM needs these on the parameter to hoist and forward region loads.
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(body->hasParamAttribute(i, llvm::Attribute::NoUndef));
    EXPECT_TRUE(body->hasParamAttribute(i, llvm::Attribute::NoFree));
    EXPECT_TRUE(body->hasParamAttribute(i, llvm::Attribute::NonNull));
    EXPECT_EQ(body->getParamAlign(i), llvm::MaybeAlign(MinAlign()));
    EXPECT_EQ(body->getParamDereferenceableBytes(i), 4);
  }
}

TEST_F(ThunkSequenceKernelEmitterTest, StoreDoesNotClobberOtherRegion) {
  constexpr absl::string_view kIncrementWord = R"(
    define ptr @increment_word(ptr %frame) {
      %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
      %args = load ptr, ptr %args_gep
      %in_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
      %in = load ptr, ptr %in_gep
      %out_gep = getelementptr %Arg, ptr %args, i32 1, i32 0
      %out = load ptr, ptr %out_gep
      %v = load i32, ptr %in
      %w = add i32 %v, 1
      store i32 %w, ptr %out
      ret ptr null
    }
  )";
  ShapedSlice x = Int(0), y = Int(64), z = Int(128);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t0,
                       Kernel("increment_word", {x}, {y}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t1,
                       Kernel("increment_word", {x}, {z}));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(t0));
  thunks.push_back(std::move(t1));
  ThunkSequence sequence = Sequence(std::move(thunks));

  ASSERT_OK_AND_ASSIGN(LlvmKernelSource source, Emit(sequence, {x}, {y, z}));
  std::vector<LlvmKernelSource> sources;
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource parsed, Parse(kIncrementWord));
  sources.push_back(std::move(parsed));
  ASSERT_OK(ThunkSequenceKernelEmitter::LinkKernels(*source.module(),
                                                    KernelModules(sources)));

  int64_t i32_loads = 0;
  IrCompiler::CompilationHooks hooks;
  hooks.post_optimization = [&](const llvm::Module& module) {
    const llvm::Function* main = module.getFunction("main");
    ASSERT_NE(main, nullptr);
    for (const llvm::BasicBlock& block : *main) {
      for (const llvm::Instruction& inst : block) {
        if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
          if (load->getType()->isIntegerTy(32)) {
            ++i32_loads;
          }
        }
      }
    }
  };
  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      llvm::TargetOptions(),
      IrCompiler::Options{llvm::CodeGenOptLevel::Default,
                          /*optimize_for_size=*/false,
                          TargetMachineOptions(GetDebugOptionsFromFlags())},
      std::move(hooks));
  ASSERT_OK_AND_ASSIGN(JitCompiler compiler,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  ASSERT_OK(compiler.AddModule(std::move(source).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(compiler).Compile(
          {FunctionLibrary::Sym<FunctionLibrary::Kernel>("main")}));
  EXPECT_EQ(i32_loads, 1);
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsTranspose) {
  Shape src_shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {2, 2}, {1, 0});
  Shape dst_shape = ShapeUtil::MakeShapeWithDenseLayout(S32, {2, 2}, {0, 1});
  BufferAllocation::Slice src(&allocation_, 0, 16), dst(&allocation_, 64, 16);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CopyThunk> copy,
      CopyThunk::Create({"copy"}, src, src_shape, dst, dst_shape));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(copy));
  ThunkSequence sequence = Sequence(std::move(thunks));
  Decision decision = ThunkSequenceKernelEmitter::CanEmit(sequence, {}, {},
                                                          Parameters(sequence));
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(), HasSubstr("Copy changes the layout"));
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsOverlappingCopy) {
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  BufferAllocation::Slice src(&allocation_, 0, 16), dst(&allocation_, 8, 16);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CopyThunk> copy,
                       CopyThunk::Create({"copy"}, src, shape, dst, shape));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(copy));
  ThunkSequence sequence = Sequence(std::move(thunks));
  Decision decision = ThunkSequenceKernelEmitter::CanEmit(
      sequence, {}, {},
      {{BufferAllocation::Slice(&allocation_, 0, 32),
        ShapeUtil::MakeShape(S32, {8})}});
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(), HasSubstr("Copy buffers overlap"));
}

TEST_F(ThunkSequenceKernelEmitterTest, CallThunkRunsNestedSequence) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence nested, SingleKernel("add", {a}, {b}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CallThunk> call,
                       CallThunk::Create({"call"}, std::move(nested)));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(call));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Value(a) = 3;
  Value(b) = 4;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {kAdd}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(b), 7);
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsUnsupportedThunkKind) {
  BufferAllocation::Slice state(&allocation_, 0, 16);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RngGetAndUpdateStateThunk> rng,
      RngGetAndUpdateStateThunk::Create({"rng"}, state, /*delta=*/1,
                                        std::make_shared<RngStateResource>()));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(rng));
  ThunkSequence sequence = Sequence(std::move(thunks));

  Decision decision = ThunkSequenceKernelEmitter::CanEmit(sequence, {}, {},
                                                          Parameters(sequence));
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(), HasSubstr("Unsupported thunk kind"));
}

TEST_F(ThunkSequenceKernelEmitterTest, ZeroSizeSlice) {
  std::string ir = absl::StrCat(R"(
    define ptr @check_empty(ptr %frame) {
      %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
      %args = load ptr, ptr %args_gep
      %e_ptr = load ptr, ptr %args
      %e_size_gep = getelementptr %Arg, ptr %args, i32 0, i32 1
      %e_size = load i64, ptr %e_size_gep
      %addr = ptrtoint ptr %e_ptr to i64
      %nonnull = icmp ne ptr %e_ptr, null
      %rem = and i64 %addr, )",
                                MinAlign() - 1, R"(
      %aligned = icmp eq i64 %rem, 0
      %zero = icmp eq i64 %e_size, 0
      %ok0 = and i1 %nonnull, %aligned
      %ok = and i1 %ok0, %zero
      %val = zext i1 %ok to i32
      %out_gep = getelementptr %Arg, ptr %args, i32 1, i32 0
      %out = load ptr, ptr %out_gep
      store i32 %val, ptr %out
      ret ptr null
    }
  )");
  ShapedSlice empty = Int(0, 0), out = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence,
                       SingleKernel("check_empty", {empty}, {out}));

  Value(out) = 0;
  ASSERT_OK_AND_ASSIGN(bool error, Run(sequence, {ir}));
  EXPECT_FALSE(error);
  EXPECT_EQ(Value(out), 1);
}

TEST_F(ThunkSequenceKernelEmitterTest, CanEmitRejectsEmptyParameterSlice) {
  ShapedSlice empty = {BufferAllocation::Slice(&allocation_, 0, 0),
                       ShapeUtil::MakeShape(S32, {0})};
  ShapedSlice out = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence,
                       SingleKernel("add", {empty}, {out}));
  Decision decision =
      ThunkSequenceKernelEmitter::CanEmit(sequence, {"add"}, {empty}, {out});
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_THAT(decision.Explain(), HasSubstr("Empty parameter slice"));
}

TEST_F(ThunkSequenceKernelEmitterTest, LinkRejectsMissingKernelDefinition) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("add", {a}, {b}));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource source,
                       Emit(sequence, Parameters(sequence)));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource decl_only,
                       Parse("declare ptr @add(ptr)\n"));
  EXPECT_THAT(ThunkSequenceKernelEmitter::LinkKernels(
                  *source.module(), {{"add", decl_only.module()}}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("does not define kernel")));
}

TEST_F(ThunkSequenceKernelEmitterTest, LinkRequiresEveryCalledKernel) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("add", {a}, {b}));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource source,
                       Emit(sequence, Parameters(sequence)));
  EXPECT_THAT(
      ThunkSequenceKernelEmitter::LinkKernels(*source.module(), {}),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("not linked: add")));
}

// Backend options apply to the module, so the linked kernel gets the union.
TEST_F(ThunkSequenceKernelEmitterTest, LinkMergesBackendOptions) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t0, Kernel("add", {a}, {b}));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Thunk> t1,
                       Kernel("lt5", {a}, {Pred(128)}));
  std::vector<std::unique_ptr<Thunk>> thunks;
  thunks.push_back(std::move(t0));
  thunks.push_back(std::move(t1));
  ThunkSequence sequence = Sequence(std::move(thunks));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource source,
                       Emit(sequence, Parameters(sequence)));

  std::vector<LlvmKernelSource> sources;
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource s0, Parse(kAdd));
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource s1, Parse(kLessThanFive));
  sources.push_back(std::move(s0));
  sources.push_back(std::move(s1));
  for (auto [source, option] :
       {std::pair{&sources[0], "b,a"}, std::pair{&sources[1], "c,a"}}) {
    source->module()->addModuleFlag(
        llvm::Module::Error, "xla_backend_extra_options",
        llvm::MDString::get(source->module()->getContext(), option));
  }
  ASSERT_OK(ThunkSequenceKernelEmitter::LinkKernels(*source.module(),
                                                    KernelModules(sources)));

  auto* options = llvm::dyn_cast_or_null<llvm::MDString>(
      source.module()->getModuleFlag("xla_backend_extra_options"));
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(options->getString(), "a,b,c");
  for (absl::string_view name : {"add", "lt5"}) {
    EXPECT_TRUE(source.module()->getFunction(name)->hasInternalLinkage());
  }
}

TEST_F(ThunkSequenceKernelEmitterTest, NestedKernelNamedBodyDoesNotCollide) {
  constexpr absl::string_view kMainBodyKernel = R"(
    define ptr @main_body(ptr %frame) {
      %args_gep = getelementptr %Frame, ptr %frame, i32 0, i32 3
      %args = load ptr, ptr %args_gep
      %in_gep = getelementptr %Arg, ptr %args, i32 0, i32 0
      %in = load ptr, ptr %in_gep
      %out_gep = getelementptr %Arg, ptr %args, i32 1, i32 0
      %out = load ptr, ptr %out_gep
      %v = load i32, ptr %in
      %w = add i32 %v, 7
      store i32 %w, ptr %out
      ret ptr null
    }
  )";
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence,
                       SingleKernel("main_body", {a}, {b}));

  Value(a) = 5;
  ASSERT_THAT(Run(sequence, {kMainBodyKernel}),
              absl_testing::IsOkAndHolds(false));
  EXPECT_EQ(Value(b), 12);
}

TEST_F(ThunkSequenceKernelEmitterTest, NestedKernelCollidingWithEntryFails) {
  ShapedSlice a = Int(0), b = Int(64);
  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence, SingleKernel("main", {a}, {b}));
  EXPECT_THAT(Emit(sequence, {a}, {b}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("collides with enclosing kernel")));
}

}  // namespace
}  // namespace xla::cpu
