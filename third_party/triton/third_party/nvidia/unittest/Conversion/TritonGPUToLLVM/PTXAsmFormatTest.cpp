#include "nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Signals.h"

#include <gtest/gtest.h>

namespace mlir {
namespace triton {
class PTXAsmFormatTest : public ::testing::Test {
protected:
  static constexpr int numValues = 4;

  PTXAsmFormatTest() {
    ctx.loadDialect<arith::ArithDialect>();

    createValues();
  }

  // Creates the test values.
  void createValues() {
    OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&block);

    // a b1 value for predicate.
    v[0] = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 1);
    for (int i = 0; i < numValues; i++) {
      v[i + 1] =
          builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), i, 32);
    }
  }

  MLIRContext ctx;
  Block block;
  Value v[numValues + 1];
};

TEST_F(PTXAsmFormatTest, basic) {
  PTXBuilder builder;

  // Create the operands needed by the instructions in the PTX code.
  auto *cst = builder.newConstantOperand(1);
  auto *val = builder.newOperand(v[1], "=r");

  // create an instruction
  auto &mov = *builder.create("mov.b16");

  mov(val, cst).predicate(v[0]);
  ASSERT_EQ(builder.dump(), "@$1 mov.b16 $0, 0x1;");

  auto values = builder.getAllMLIRArgs();
  ASSERT_EQ(values[0], v[1]); // $0 -> v[1]
  ASSERT_EQ(values[1], v[0]); // $1 -> v[0]

  auto constraints = builder.getConstraints();
  ASSERT_EQ(constraints, "=r,b"); // $0 -> =r, $1 -> b
}

TEST_F(PTXAsmFormatTest, complexInstruction) {
  using triton::CacheModifier;
  using triton::EvictionPolicy;

  PTXBuilder builder;

  int width = 16;
  int nWords = 2;

  Value predicateVal = v[0];
  Value addrVal = v[1];

  auto addr = builder.newAddrOperand(addrVal, "l", 128 /*offset*/);

  bool isVolatile = false;
  auto cache = triton::CacheModifier::CA;
  auto cachePriority = triton::EvictionPolicy::EVICT_FIRST;
  bool hasL2EvictPolicy = true;

  auto &ld =
      builder
          .create<>("ld") //
          ->o("volatile", isVolatile)
          .global()
          .o("ca", cache == CacheModifier::CA)
          .o("cg", cache == CacheModifier::CG)
          .o("L1::evict_first", cachePriority == EvictionPolicy::EVICT_FIRST)
          .o("L1::evict_last", cachePriority == EvictionPolicy::EVICT_LAST)
          .o("L1::cache_hint", hasL2EvictPolicy)
          .v(nWords)
          .b(width);

  // Link the instruction to operands
  ld(addr).predicate(predicateVal);

  EXPECT_EQ(
      builder.dump(),
      "@$1 ld.global.ca.L1::evict_first.L1::cache_hint.v2.b16 [ $0 + 128 ];");
  auto values = builder.getAllMLIRArgs();
  EXPECT_EQ(values[0], addrVal);      // $0 -> predicate
  EXPECT_EQ(values[1], predicateVal); // $1 -> addr
  EXPECT_EQ(builder.getConstraints(), "l,b");
}

TEST_F(PTXAsmFormatTest, MultiLinePTX) {
  PTXBuilder builder;

  auto *constVal = builder.newConstantOperand(1);
  auto *valVal0 = builder.newOperand(v[1], "=r");
  auto *valVal1 = builder.newOperand(v[2], "=r");

  auto &mov = *builder.create("mov");

  mov(valVal0, constVal);
  mov(valVal1, constVal);
  mov(valVal1, valVal0);

  EXPECT_EQ(builder.dump(), "mov $0, 0x1;\n\t"
                            "mov $1, 0x1;\n\t"
                            "mov $1, $0;");

  auto values = builder.getAllMLIRArgs();
  EXPECT_EQ(values[0], v[1]); // $0 -> v[1]
  EXPECT_EQ(values[1], v[2]); // $1 -> v[2]
}

TEST_F(PTXAsmFormatTest, onlyAttachMLIRArgs) {
  PTXBuilder builder;
  const char *ptxCode =
      ".param .b64 param0;\n" // prepare param0 (format string)
      "st.param.b64 [param0], %0;\n"
      "st.param.b64 [param0], %1;\n"
      "st.param.b64 [param0], %2;\n";

  auto &ptxSnippet = *builder.create(ptxCode);
  auto *opr0 = builder.newOperand(v[0], "r");
  auto *opr1 = builder.newOperand(v[1], "r");
  auto *opr2 = builder.newOperand(v[2], "r");
  ptxSnippet({opr1, opr2, opr0}, true);

  EXPECT_EQ(builder.dump(), ptxCode);
  ASSERT_EQ(builder.getAllMLIRArgs()[0], v[1]);
  ASSERT_EQ(builder.getAllMLIRArgs()[1], v[2]);
  ASSERT_EQ(builder.getAllMLIRArgs()[2], v[0]);
  ASSERT_EQ(builder.getAllMLIRArgs().size(), 3);
}

} // namespace triton
} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
