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

#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_16_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_64_ll.h"
#include "xla/codegen/intrinsic/intrinsic_compiler_lib.h"
#include "xla/codegen/intrinsic/simple_jit_runner.h"
#include "xla/codegen/intrinsic/test_matchers.h"
#include "tsl/platform/denormal.h"

namespace xla::codegen {
namespace {

using ::xla::codegen::intrinsic::JitRunner;
using ::xla::codegen::intrinsic::NearUlps;

llvm::FunctionType* UnaryType(llvm::Type* type) {
  return llvm::FunctionType::get(type, {type}, /*isVarArg=*/false);
}

llvm::Type* VecF32(llvm::LLVMContext& context, int width) {
  return llvm::VectorType::get(llvm::Type::getFloatTy(context),
                               llvm::ElementCount::getFixed(width));
}

void StripHostAttrs(llvm::Module& module) {
  for (llvm::Function& f : module) {
    f.removeFnAttr("probe-stack");
    f.removeFnAttr("target-cpu");
    f.removeFnAttr("target-features");
  }
}

TEST(CppGenIntrinsicsTest, AtanV8F32HasDirectSignature) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary16LlIr);
  llvm::FunctionType* type = UnaryType(VecF32(context, 8));

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.atan.v8f32", type);
  ASSERT_NE(fn, nullptr);
  EXPECT_EQ(fn->getFunctionType(), type);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
}

TEST(CppGenIntrinsicsTest, AtanV8F32IsIdempotent) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary16LlIr);
  llvm::FunctionType* type = UnaryType(VecF32(context, 8));

  llvm::Function* first =
      GetCppGenFunction(module.get(), "xla.atan.v8f32", type);
  llvm::Function* second =
      GetCppGenFunction(module.get(), "xla.atan.v8f32", type);
  EXPECT_EQ(first, second);
  EXPECT_EQ(second->getFunctionType(), type);
}

TEST(CppGenIntrinsicsTest, DirectSignaturesAreUntouched) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary16LlIr);

  llvm::Function* scalar = GetCppGenFunction(
      module.get(), "xla.atan.f32", UnaryType(llvm::Type::getFloatTy(context)));
  EXPECT_EQ(scalar, module->getFunction("xla.atan.f32"));
  EXPECT_EQ(module->getFunction("xla.atan.f32.body"), nullptr);

  llvm::Function* v4 = GetCppGenFunction(module.get(), "xla.atan.v4f32",
                                         UnaryType(VecF32(context, 4)));
  EXPECT_EQ(v4, module->getFunction("xla.atan.v4f32"));
  EXPECT_EQ(module->getFunction("xla.atan.v4f32.body"), nullptr);
}

TEST(CppGenIntrinsicsTest, AtanV8F32ComputesThroughJit) {
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, llvm_ir::kEigenUnary16LlIr);
  StripHostAttrs(*module);

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.atan.v8f32",
                                         UnaryType(VecF32(*context, 8)));
  fn->setLinkage(llvm::Function::ExternalLinkage);
  std::string name = fn->getName().str();

  JitRunner jit(std::move(module), std::move(context));
  auto atan = jit.GetVectorizedFn<8, float, float>(name);

  std::array<float, 8> x = {0.0f, 0.5f, -1.5f, 3.0f, 0.4f, 2.0f, -4.0f, 5.5f};
  std::array<float, 8> y = atan(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(y[i], std::atan(x[i]), 1e-6f) << "lane " << i;
  }
}

// Eigen's vector atan is up to 2 ulp off, e.g. float atan(0.5).
constexpr int kAtanUlps = 2;

// Applies xla.atan from `ir_bitcode`, obtained through the same
// GetCppGenFunction path kernels use, to `xs` in <N x T> chunks.
template <typename T, size_t N>
std::vector<T> AtanFromIr(const std::string& ir_bitcode,
                          const std::vector<T>& xs,
                          bool flush_denormals = false) {
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, ir_bitcode);
  StripHostAttrs(*module);
  std::string name =
      absl::StrCat("xla.atan.v", N, sizeof(T) == 4 ? "f32" : "f64");
  llvm::Type* type = llvm::VectorType::get(llvm::Type::getScalarTy<T>(*context),
                                           llvm::ElementCount::getFixed(N));
  llvm::Function* fn = GetCppGenFunction(module.get(), name, UnaryType(type));
  EXPECT_TRUE(fn->hasFnAttribute(llvm::Attribute::AlwaysInline)) << name;
  fn->setLinkage(llvm::Function::ExternalLinkage);

  JitRunner jit(std::move(module), std::move(context));
  auto atan = jit.GetVectorizedFn<N, T, T>(name);
  std::optional<tsl::port::ScopedFlushDenormal> flush;
  if (flush_denormals) {
    flush.emplace();
  }
  std::vector<T> ys;
  for (size_t i = 0; i < xs.size(); i += N) {
    size_t n = std::min(N, xs.size() - i);
    std::array<T, N> x = {};
    std::copy_n(xs.begin() + i, n, x.begin());
    std::array<T, N> y = atan(x);
    ys.insert(ys.end(), y.begin(), y.begin() + n);
  }
  return ys;
}

template <typename T, size_t N>
void ExpectAtanNearStd(const std::string& ir_bitcode,
                       const std::vector<T>& xs) {
  std::vector<T> ys = AtanFromIr<T, N>(ir_bitcode, xs);
  for (size_t i = 0; i < xs.size(); ++i) {
    EXPECT_THAT(ys[i], NearUlps(std::atan(xs[i]), kAtanUlps)) << xs[i];
  }
}

template <typename T, size_t N>
void ExpectAtan16NearStd(const std::vector<T>& xs) {
  ExpectAtanNearStd<T, N>(llvm_ir::kEigenUnary16LlIr, xs);
}

// x86-64 SysV without AVX-512 and AArch64 pass 512-bit vectors indirectly, so
// these go through CreateDirectAdapter. The Windows x64 ABI that clang-cl emits
// IR for passes them by value, so there is nothing to adapt there.
TEST(CppGenIntrinsicsTest, WideAtanIsAdaptedFromLibraryIr) {
  {
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module =
        ParseEmbeddedBitcode(context, llvm_ir::kEigenUnary16LlIr);
    if (!module->getTargetTriple().isOSWindows()) {
      for (const char* name : {"xla.atan.v16f32", "xla.atan.v8f64"}) {
        EXPECT_TRUE(
            module->getFunction(name)->getArg(0)->getType()->isPointerTy())
            << name;
      }
    }
  }
  ExpectAtan16NearStd<float, 16>({0.0f, 0.5f, -1.0f, 1.5f, -2.0f, 3.0f, 0.1f,
                                  -0.25f, 10.0f, -40.0f, 1e-4f, -0.9f, 7.0f,
                                  -1e3f, 0.75f, -5.5f});
  ExpectAtan16NearStd<double, 8>({0.0, 0.5, -1.0, 1.5, -2.0, 3.0, 1e-4, -1e3});
}

TEST(CppGenIntrinsicsTest, EigenIntrinsicsAreAvailable) {
  EXPECT_TRUE(AreEigenIntrinsicsAvailable());
  EXPECT_TRUE(AreCppGenIntrinsicsAvailable());
}

TEST(CppGenIntrinsicsTest, AtanV2F64FromLibraryIr) {
  ExpectAtan16NearStd<double, 2>({0.5, -3.0, 1.0, -0.1, 1e-4, 1e3});
}

TEST(CppGenIntrinsicsTest, SubPacketAtanFromWideLibraryIr) {
  const std::vector<float> xs_f32 = {0.0f,  0.5f,   -1.0f, 1.5f,
                                     -2.0f, 3.0f,   1e-4f, -0.25f,
                                     10.0f, -40.0f, 0.75f, -5.5f};
  const std::vector<double> xs_f64 = {0.0,  0.5, -1.0, 1.5,
                                      -2.0, 3.0, 1e-4, -1e3};

  ExpectAtanNearStd<float, 4>(llvm_ir::kEigenUnary32LlIr, xs_f32);
  ExpectAtanNearStd<float, 8>(llvm_ir::kEigenUnary32LlIr, xs_f32);
  ExpectAtanNearStd<double, 2>(llvm_ir::kEigenUnary32LlIr, xs_f64);
  ExpectAtanNearStd<double, 4>(llvm_ir::kEigenUnary32LlIr, xs_f64);

  ExpectAtanNearStd<float, 4>(llvm_ir::kEigenUnary64LlIr, xs_f32);
  ExpectAtanNearStd<float, 8>(llvm_ir::kEigenUnary64LlIr, xs_f32);
  ExpectAtanNearStd<double, 2>(llvm_ir::kEigenUnary64LlIr, xs_f64);
  ExpectAtanNearStd<double, 4>(llvm_ir::kEigenUnary64LlIr, xs_f64);
}

template <typename T, size_t N>
void ExpectAtanEdgeCases(const std::string& ir_bitcode, bool flush_denormals) {
  using Limits = std::numeric_limits<T>;
  const T kShortcut = T(1e-3);
  const std::vector<T> xs = {T(0),
                             -T(0),
                             Limits::denorm_min(),
                             Limits::min() / 1024,
                             -Limits::min() / 1024,
                             Limits::quiet_NaN(),
                             Limits::infinity(),
                             -Limits::infinity(),
                             std::nextafter(kShortcut, T(0)),
                             std::nextafter(kShortcut, T(1)),
                             -std::nextafter(kShortcut, T(0)),
                             -std::nextafter(kShortcut, T(1)),
                             T(0.5),
                             T(-3)};
  std::vector<T> ys = AtanFromIr<T, N>(ir_bitcode, xs, flush_denormals);
  for (size_t i = 0; i < xs.size(); ++i) {
    T x = xs[i];
    T y = ys[i];
    if (flush_denormals && std::fpclassify(x) == FP_SUBNORMAL) {
      // DAZ may read a subnormal input as zero.
      EXPECT_TRUE(y == 0 || y == std::atan(x)) << x << " -> " << y;
    } else {
      EXPECT_THAT(y, NearUlps(std::atan(x), kAtanUlps)) << x;
    }
    if (!std::isnan(x)) {
      EXPECT_EQ(std::signbit(y), std::signbit(x)) << x << " -> " << y;
    }
  }
}

TEST(CppGenIntrinsicsTest, AtanEdgeCasesFromLibraryIr) {
  for (const std::string* ir :
       {&llvm_ir::kEigenUnary16LlIr, &llvm_ir::kEigenUnary32LlIr,
        &llvm_ir::kEigenUnary64LlIr}) {
    for (bool flush_denormals : {false, true}) {
      SCOPED_TRACE(absl::StrCat("flush_denormals=", flush_denormals));
      ExpectAtanEdgeCases<float, 4>(*ir, flush_denormals);
      ExpectAtanEdgeCases<double, 2>(*ir, flush_denormals);
    }
  }
}

// x86-64 without AVX passes 256-bit vectors as byval pointers.
TEST(CppGenIntrinsicsTest, ByvalArgumentIsAdapted) {
  constexpr absl::string_view kIr = R"(
    define <8 x float> @xla.neg.v8f32(ptr byval(<8 x float>) align 16 %p) {
      %x = load <8 x float>, ptr %p, align 16
      %r = fneg <8 x float> %x
      ret <8 x float> %r
    }
  )";
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kIr));
  llvm::FunctionType* type = UnaryType(VecF32(*context, 8));

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.neg.v8f32", type);
  EXPECT_EQ(fn->getFunctionType(), type);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
  fn->setLinkage(llvm::Function::ExternalLinkage);

  JitRunner jit(std::move(module), std::move(context));
  auto neg = jit.GetVectorizedFn<8, float, float>("xla.neg.v8f32");
  std::array<float, 8> x = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::array<float, 8> y = neg(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(y[i], -x[i]) << "lane " << i;
  }
}

// AArch64 returns vectors wider than 128 bits through sret and passes them
// through an unannotated pointer.
TEST(CppGenIntrinsicsTest, SretReturnIsAdapted) {
  constexpr absl::string_view kIr = R"(
    define void @xla.neg.v8f32(ptr sret(<8 x float>) align 16 %out, ptr %p) {
      %x = load <8 x float>, ptr %p, align 16
      %r = fneg <8 x float> %x
      store <8 x float> %r, ptr %out, align 16
      ret void
    }
  )";
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kIr));
  llvm::FunctionType* type = UnaryType(VecF32(*context, 8));

  llvm::Function* fn = GetCppGenFunction(module.get(), "xla.neg.v8f32", type);
  EXPECT_EQ(fn->getFunctionType(), type);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
  fn->setLinkage(llvm::Function::ExternalLinkage);

  JitRunner jit(std::move(module), std::move(context));
  auto neg = jit.GetVectorizedFn<8, float, float>("xla.neg.v8f32");
  std::array<float, 8> x = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::array<float, 8> y = neg(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(y[i], -x[i]) << "lane " << i;
  }
}

class LinkTest : public ::testing::TestWithParam<absl::string_view> {};

TEST_P(LinkTest, AdaptsMismatchedDeclaration) {
  constexpr absl::string_view kLib = R"(
    define void $0(ptr sret(<8 x float>) align 16 %out, ptr %p) {
      %x = load <8 x float>, ptr %p, align 16
      %r = fneg <8 x float> %x
      store <8 x float> %r, ptr %out, align 16
      ret void
    }
  )";
  constexpr absl::string_view kKernel = R"(
    declare <8 x float> @xla.neg.v8f32(<8 x float>)
    define <8 x float> @kernel(<8 x float> %x) {
      %r = call <8 x float> @xla.neg.v8f32(<8 x float> %x)
      ret <8 x float> %r
    }
  )";
  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kKernel));
  CppGenIntrinsicLibrary(absl::Substitute(kLib, GetParam()), "lib")
      .LinkIntoModule(*module);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));

  llvm::Function* kernel = module->getFunction("kernel");
  auto* call = llvm::cast<llvm::CallInst>(&kernel->getEntryBlock().front());
  llvm::Function* callee = call->getCalledFunction();
  ASSERT_NE(callee, nullptr);
  ASSERT_FALSE(callee->isDeclaration());
  EXPECT_EQ(callee->getName(), "xla.neg.v8f32");
  EXPECT_EQ(module->getFunction("xla.neg.v8f32.old_decl"), nullptr);
  for (const llvm::Function& f : *module) {
    EXPECT_FALSE(f.getName().starts_with("\01")) << f.getName().str();
  }

  JitRunner jit(std::move(module), std::move(context));
  auto neg = jit.GetVectorizedFn<8, float, float>("kernel");
  std::array<float, 8> x = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  std::array<float, 8> y = neg(x);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(y[i], -x[i]) << "lane " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(CppGenIntrinsicsTest, LinkTest,
                         ::testing::Values("@xla.neg.v8f32",
                                           R"(@"\01xla.neg.v8f32")"));

// When bitcode compiled for baseline SysV x86-64 (+sse2, passing <4 x double>
// via ptr byval) is linked into a kernel calling @xla.atan.v4f64(<4 x double>)
// by value, LinkIntoModule + RunInlineAndOptPasses must adapt the ABI, strip
// host target and stack-protector attributes, and fully inline the body.
TEST(CppGenIntrinsicsTest, ByvalAtanV4F64InlinesIntoCaller) {
  constexpr absl::string_view kLib = R"(
    define <4 x double> @xla.atan.v4f64(ptr nocapture readonly byval(<4 x double>) align 32 %p) #0 {
      %x = load <4 x double>, ptr %p, align 32
      %r = fneg <4 x double> %x
      ret <4 x double> %r
    }

    attributes #0 = { memory(argmem: read) nounwind ssp uwtable "frame-pointer"="all" "min-legal-vector-width"="256" "prefer-vector-width"="512" "probe-stack"="inline-asm" "stack-probe-size"="4096" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
  )";
  constexpr absl::string_view kKernel = R"(
    declare <4 x double> @xla.atan.v4f64(<4 x double>)

    define <4 x double> @kernel(<4 x double> %x) {
      %r = call <4 x double> @xla.atan.v4f64(<4 x double> %x)
      ret <4 x double> %r
    }
  )";

  auto context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(*context, std::string(kKernel));
  CppGenIntrinsicLibrary(std::string(kLib), "lib").LinkIntoModule(*module);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));

  llvm::Function* adapter = module->getFunction("xla.atan.v4f64");
  ASSERT_NE(adapter, nullptr);
  EXPECT_TRUE(adapter->hasFnAttribute(llvm::Attribute::AlwaysInline));
  EXPECT_FALSE(adapter->hasFnAttribute("target-cpu"));
  EXPECT_FALSE(adapter->hasFnAttribute("target-features"));
  EXPECT_FALSE(adapter->hasFnAttribute(llvm::Attribute::StackProtect));

  llvm::Function* body = module->getFunction("xla.atan.v4f64.body");
  ASSERT_NE(body, nullptr);
  EXPECT_FALSE(body->hasFnAttribute("target-cpu"));
  EXPECT_FALSE(body->hasFnAttribute("target-features"));
  EXPECT_FALSE(body->hasFnAttribute("probe-stack"));
  EXPECT_FALSE(body->hasFnAttribute("stack-probe-size"));
  EXPECT_FALSE(body->hasFnAttribute("stack-protector-buffer-size"));
  EXPECT_FALSE(body->hasFnAttribute(llvm::Attribute::StackProtect));

  intrinsic::RunInlineAndOptPasses(*module);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));
  EXPECT_EQ(module->getFunction("xla.atan.v4f64"), nullptr);
  EXPECT_EQ(module->getFunction("xla.atan.v4f64.body"), nullptr);

  llvm::Function* kernel = module->getFunction("kernel");
  ASSERT_NE(kernel, nullptr);
  EXPECT_FALSE(kernel->hasFnAttribute("target-cpu"));
  EXPECT_FALSE(kernel->hasFnAttribute("target-features"));
  EXPECT_FALSE(kernel->hasFnAttribute("probe-stack"));
  EXPECT_FALSE(kernel->hasFnAttribute("stack-probe-size"));
  EXPECT_FALSE(kernel->hasFnAttribute("stack-protector-buffer-size"));
  EXPECT_FALSE(kernel->hasFnAttribute(llvm::Attribute::StackProtect));
  for (const llvm::BasicBlock& bb : *kernel) {
    for (const llvm::Instruction& inst : bb) {
      EXPECT_FALSE(llvm::isa<llvm::CallInst>(inst));
      EXPECT_FALSE(llvm::isa<llvm::AllocaInst>(inst));
    }
  }

  JitRunner jit(std::move(module), std::move(context));
  auto fn = jit.GetVectorizedFn<4, double, double>("kernel");
  std::array<double, 4> x = {1.0, -2.0, 3.5, -0.25};
  std::array<double, 4> y = fn(x);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(y[i], -x[i]) << "lane " << i;
  }
}

}  // namespace
}  // namespace xla::codegen
