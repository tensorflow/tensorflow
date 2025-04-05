#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_UTILITY_H

#include <set>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "ttgpu_to_llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace mlir::LLVM {
using namespace mlir::triton;

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v);
Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v);
Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);
Value createConstantF16(Location loc, OpBuilder &rewriter, float v);
Value createConstantBF16(Location loc, OpBuilder &rewriter, float v);
Value createConstantF32(Location loc, OpBuilder &rewriter, float v);
Value createConstantF64(Location loc, OpBuilder &rewriter, double v);
Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type);
Value createIndexConstant(OpBuilder &builder, Location loc,
                          const TypeConverter *converter, int64_t value);
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value);

LLVM::CallOp createLLVMCallOp(OpBuilder &builder, Location loc,
                              LLVMFuncOp funcOp, ValueRange args);
LLVM::CallIntrinsicOp
createLLVMIntrinsicCallOp(OpBuilder &builder, Location loc, StringRef intrinsic,
                          TypeRange types, ValueRange args);
} // namespace mlir::LLVM

// Is v an integer or floating-point scalar constant equal to 0?
bool isConstantZero(Value v);

namespace mlir::triton {

struct TritonLLVMOpBuilder {
  TritonLLVMOpBuilder(Location loc, OpBuilder &builder)
      : loc(loc), builder(&builder) {}

  // Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
  // Operators
  template <typename... Args> LLVM::SIToFPOp inttofloat(Args &&...args) {
    return builder->create<LLVM::SIToFPOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::IntToPtrOp inttoptr(Args &&...args) {
    return builder->create<LLVM::IntToPtrOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::PtrToIntOp ptrtoint(Args &&...args) {
    return builder->create<LLVM::PtrToIntOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ZExtOp zext(Args &&...args) {
    return builder->create<LLVM::ZExtOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::SExtOp sext(Args &&...args) {
    return builder->create<LLVM::SExtOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::FPExtOp fpext(Args &&...args) {
    return builder->create<LLVM::FPExtOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::FPTruncOp fptrunc(Args &&...args) {
    return builder->create<LLVM::FPTruncOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::TruncOp trunc(Args &&...args) {
    return builder->create<LLVM::TruncOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::UDivOp udiv(Args &&...args) {
    return builder->create<LLVM::UDivOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::SDivOp sdiv(Args &&...args) {
    return builder->create<LLVM::SDivOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::URemOp urem(Args &&...args) {
    return builder->create<LLVM::URemOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::AddOp add(Args &&...args) {
    return builder->create<LLVM::AddOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::SubOp sub(Args &&...args) {
    return builder->create<LLVM::SubOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::FAddOp fadd(Args &&...args) {
    return builder->create<LLVM::FAddOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::MulOp mul(Args &&...args) {
    return builder->create<LLVM::MulOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::FMulOp fmul(Args &&...args) {
    return builder->create<LLVM::FMulOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::FMAOp fma(Args &&...args) {
    return builder->create<LLVM::FMAOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::FNegOp neg(Args &&...args) {
    return builder->create<LLVM::FNegOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::SMaxOp smax(Args &&...args) {
    return builder->create<LLVM::SMaxOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::UMaxOp umax(Args &&...args) {
    return builder->create<LLVM::UMaxOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::MaxNumOp fmax(Args &&...args) {
    return builder->create<LLVM::MaxNumOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::SMinOp smin(Args &&...args) {
    return builder->create<LLVM::SMinOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::UMinOp umin(Args &&...args) {
    return builder->create<LLVM::UMinOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::MinNumOp fmin(Args &&...args) {
    return builder->create<LLVM::MinNumOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ShlOp shl(Args &&...args) {
    return builder->create<LLVM::ShlOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::LShrOp lshr(Args &&...args) {
    return builder->create<LLVM::LShrOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::AShrOp ashr(Args &&...args) {
    return builder->create<LLVM::AShrOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::AndOp and_(Args &&...args) {
    return builder->create<LLVM::AndOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::XOrOp xor_(Args &&...args) {
    return builder->create<LLVM::XOrOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::OrOp or_(Args &&...args) {
    return builder->create<LLVM::OrOp>(loc, std::forward<Args>(args)...);
  }
  LLVM::BitcastOp bitcast(Value val, Type type) {
    return builder->create<LLVM::BitcastOp>(loc, type, val);
  }
  template <typename... Args>
  LLVM::AddrSpaceCastOp addrspacecast(Args &&...args) {
    return builder->create<LLVM::AddrSpaceCastOp>(loc,
                                                  std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::GEPOp gep(Args &&...args) {
    return builder->create<LLVM::GEPOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::InsertValueOp insert_val(Args &&...args) {
    return builder->create<LLVM::InsertValueOp>(loc,
                                                std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ExtractValueOp extract_val(Args &&...args) {
    return builder->create<LLVM::ExtractValueOp>(loc,
                                                 std::forward<Args>(args)...);
  }
  template <typename... Args>
  LLVM::InsertElementOp insert_element(Args &&...args) {
    return builder->create<LLVM::InsertElementOp>(loc,
                                                  std::forward<Args>(args)...);
  }
  template <typename... Args>
  LLVM::ExtractElementOp extract_element(Args &&...args) {
    return builder->create<LLVM::ExtractElementOp>(loc,
                                                   std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::LoadOp load(Args &&...args) {
    return builder->create<LLVM::LoadOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::StoreOp store(Args &&...args) {
    return builder->create<LLVM::StoreOp>(loc, std::forward<Args>(args)...);
  }
  LLVM::FCmpOp fcmp_ogt(Value lhs, Value rhs) {
    return builder->create<LLVM::FCmpOp>(loc, builder->getI1Type(),
                                         LLVM::FCmpPredicate::ogt, lhs, rhs);
  }
  LLVM::FCmpOp fcmp_olt(Value lhs, Value rhs) {
    return builder->create<LLVM::FCmpOp>(loc, builder->getI1Type(),
                                         LLVM::FCmpPredicate::olt, lhs, rhs);
  }
  LLVM::FCmpOp fcmp_eq(Value lhs, Value rhs) {
    return builder->create<LLVM::FCmpOp>(loc, builder->getI1Type(),
                                         LLVM::FCmpPredicate::oeq, lhs, rhs);
  }
  template <typename... Args> LLVM::ICmpOp icmp_eq(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_ne(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_slt(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_sle(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_sgt(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_sge(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_ult(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_ule(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ule,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_ugt(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ICmpOp icmp_uge(Args &&...args) {
    return builder->create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge,
                                         std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::SelectOp select(Args &&...args) {
    return builder->create<LLVM::SelectOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::AddressOfOp address_of(Args &&...args) {
    return builder->create<LLVM::AddressOfOp>(loc, std::forward<Args>(args)...);
  }
  mlir::gpu::BarrierOp barrier() {
    return builder->create<mlir::gpu::BarrierOp>(loc);
  }
  template <typename... Args> LLVM::UndefOp undef(Args &&...args) {
    return builder->create<LLVM::UndefOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::ZeroOp null(Args &&...args) {
    return builder->create<LLVM::ZeroOp>(loc, std::forward<Args>(args)...);
  }
  template <typename... Args> LLVM::CallOp call(Args &&...args) {
    return builder->create<LLVM::CallOp>(loc, std::forward<Args>(args)...);
  }
  // Constants
  Value int_val(short bitwidth, int64_t val) {
    Type ty = builder->getIntegerType(bitwidth);
    return builder->create<LLVM::ConstantOp>(loc, ty,
                                             builder->getIntegerAttr(ty, val));
  }
  Value i1_val(int64_t val) { return int_val(1, val); }
  Value true_val() { return int_val(1, true); }
  Value false_val() { return int_val(1, false); }
  Value f16_val(float v) { return LLVM::createConstantF16(loc, *builder, v); }
  Value bf16_val(float v) { return LLVM::createConstantBF16(loc, *builder, v); }
  Value f32_val(float v) { return LLVM::createConstantF32(loc, *builder, v); }
  Value f64_val(double v) { return LLVM::createConstantF64(loc, *builder, v); }
  Value i8_val(int64_t val) { return int_val(8, val); }
  Value i16_val(int64_t val) { return int_val(16, val); }
  Value i32_val(int64_t val) { return int_val(32, val); }
  Value i64_val(int64_t val) { return int_val(64, val); }

  Location loc;
  OpBuilder *builder;
};

// This builder combines an IRRewriter and a TritonLLVMOpBuilder into one,
// making it easy to create operations with an implicit location and create LLVM
// operations with shorthands.
class TritonLLVMIRRewriter : public IRRewriter, public TritonLLVMOpBuilder {
public:
  // Create a builder with an implicit location. Arguments are forwarded to
  // IRRewriter's constructor.
  template <typename... Args>
  TritonLLVMIRRewriter(Location loc, Args &&...args)
      : IRRewriter(std::forward<Args>(args)...),
        TritonLLVMOpBuilder(loc, *this) {}

  // Get the implicit location.
  Location getLoc() const { return loc; }
  // Set the implicit location used to build ops.
  void setLoc(Location loc) { this->loc = loc; }

  // Wrapper for op creation that passes an implicit location.
  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    return OpBuilder::create<OpTy>(loc, std::forward<Args>(args)...);
  }
};
} // namespace mlir::triton

// Types
#define ptr_ty(...) LLVM::LLVMPointerType::get(__VA_ARGS__)
#define int_ty(width) rewriter.getIntegerType(width)
#define i64_ty rewriter.getIntegerType(64)
#define i32_ty rewriter.getIntegerType(32)
#define i16_ty rewriter.getIntegerType(16)
#define i32_ty rewriter.getIntegerType(32)
#define i64_ty rewriter.getIntegerType(64)
#define ui32_ty rewriter.getIntegerType(32, false)
#define ui64_ty rewriter.getIntegerType(64, false)
#define f16_ty rewriter.getF16Type()
#define bf16_ty rewriter.getBF16Type()
#define i8_ty rewriter.getIntegerType(8)
#define i1_ty rewriter.getI1Type()
#define f32_ty rewriter.getF32Type()
#define f64_ty rewriter.getF64Type()
#define vec_ty(type, num) VectorType::get(num, type)
#define void_ty(ctx) LLVM::LLVMVoidType::get(ctx)
#define struct_ty(...) LLVM::LLVMStructType::getLiteral(ctx, __VA_ARGS__)
#define array_ty(elemTy, count) LLVM::LLVMArrayType::get(elemTy, count)

// Attributes
#define i32_arr_attr(...) rewriter.getI32ArrayAttr({__VA_ARGS__})
#define i64_arr_attr(...) rewriter.getI64ArrayAttr({__VA_ARGS__})
#define str_attr(str) ::mlir::StringAttr::get(ctx, (str))

namespace mlir {
namespace triton {

namespace gpu {
Type getFunctionType(Type resultType, ValueRange operands);

LLVM::LLVMFuncOp appendOrGetExternFuncOp(RewriterBase &rewriter, Operation *op,
                                         StringRef funcName, Type funcType,
                                         StringRef libname = "",
                                         StringRef libpath = "");
} // namespace gpu

} // namespace triton

namespace LLVM {
using namespace mlir::triton;

// Is v an integer or floating-point scalar constant equal to 0?
bool isConstantZero(Value v);

class SharedMemoryObject {
public:
  SharedMemoryObject(Value base, Type baseElemType, ArrayRef<Value> offsets)
      : base(base), baseElemType(baseElemType),
        offsets(offsets.begin(), offsets.end()) {}

  SharedMemoryObject(Value base, Type baseElemType, int64_t rank, Location loc,
                     RewriterBase &rewriter)
      : base(base), baseElemType(baseElemType) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    offsets.append(rank, b.i32_val(0));
  }

  SmallVector<Value> getOffsets() const { return offsets; }
  Value getBase() const { return base; }
  Type getBaseElemType() const { return baseElemType; }

  SmallVector<Value> getElems() const {
    SmallVector<Value> elems;
    elems.push_back(base);
    elems.append(offsets.begin(), offsets.end());
    return elems;
  }

  SmallVector<Type> getTypes() const {
    SmallVector<Type> types;
    types.push_back(base.getType());
    types.append(offsets.size(), IntegerType::get(base.getContext(), 32));
    return types;
  }

  SmallVector<Value> getStrides(triton::gpu::MemDescType memDesc, Location loc,
                                RewriterBase &rewriter) const {
    auto allocShape = memDesc.getAllocShape();
    auto allocShapePerCTA = triton::gpu::getAllocationShapePerCTA(
        memDesc.getEncoding(), allocShape);
    auto layoutOrder = triton::gpu::getOrder(memDesc);
    auto allocStrides = SharedMemoryObject::getStridesForShape(
        allocShapePerCTA, layoutOrder, loc, rewriter);
    return SmallVector<Value>(allocStrides.end() - offsets.size(),
                              allocStrides.end());
  }

  // TODO(Keren): deprecate the method once AMD backend has cleaned up
  Value getCSwizzleOffset(int dim) const {
    assert(dim >= 0 && dim < offsets.size());
    return offsets[dim];
  }

  // TODO(Keren): deprecate the method once AMD backend has cleaned up
  Value getBaseBeforeSlice(int dim, Location loc,
                           RewriterBase &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value cSwizzleOffset = getCSwizzleOffset(dim);
    Value offset = b.sub(b.i32_val(0), cSwizzleOffset);
    Type type = base.getType();
    return b.gep(type, baseElemType, base, offset);
  }

private:
  static SmallVector<unsigned>
  getOrderForShape(ArrayRef<int64_t> shape, ArrayRef<unsigned> layoutOrder) {
    SmallVector<unsigned> order(shape.size());
    // Default minor-to-major order
    std::iota(order.rbegin(), order.rend(), 0);
    if (layoutOrder.size() > 0) {
      // If a layout order is provided, we assume it specifies the order in
      // which the dimensions are first accessed, and unspecified dimensions
      // retain the minor-to-major order. For example, if order = [2, 1, 0] and
      // layoutOrder = [0, 1], we need to shift `layoutOrder`
      // by -1 (move them right). The resulting order will then be [1, 2, 0].
      int rankDiff = layoutOrder.size() - shape.size();
      auto minRank = std::min<size_t>(shape.size(), layoutOrder.size());
      for (size_t i = 0; i < minRank; ++i)
        order[i] = layoutOrder[i] - rankDiff;
    }
    assert(isPermutationOfIota(order) && "Invalid order");
    return order;
  }

  static SmallVector<Value> getStridesForShape(ArrayRef<int64_t> shape,
                                               ArrayRef<unsigned> layoutOrder,
                                               Location loc,
                                               RewriterBase &rewriter) {
    SmallVector<Value> strides(shape.size());
    auto order = SharedMemoryObject::getOrderForShape(shape, layoutOrder);
    int64_t stride = 1;
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (auto idx : order) {
      strides[idx] = b.i32_val(stride);
      stride *= shape[idx];
    }
    return strides;
  }

  Value base; // i32 ptr. The start address of the shared memory object.
  Type baseElemType;
  SmallVector<Value>
      offsets; // i32 int. The offsets are zero at the initial allocation.
};

Value getStructFromSharedMemoryObject(Location loc,
                                      const SharedMemoryObject &smemObj,
                                      RewriterBase &rewriter);

SharedMemoryObject getSharedMemoryObjectFromStruct(Location loc,
                                                   Value llvmStruct,
                                                   Type elemTy,
                                                   RewriterBase &rewriter);

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order);

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape);

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape);

SmallVector<unsigned> delinearize(unsigned linear, ArrayRef<unsigned> shape,
                                  ArrayRef<unsigned> order);

// Returns a tuple with the delinearized coordinates and a boolean which is true
// iff the Value is not broadcasted (equivalently, if the value is the "first"
// lane/thread/etc. that holds the given value). In mathy terms, the boolean is
// true if the element is the canonical representative of the class.
std::tuple<SmallVector<Value>, Value>
delinearize(RewriterBase &rewriter, Location loc,
            triton::gpu::DistributedEncodingTrait layout,
            ArrayRef<int64_t> shape, StringAttr dimName, Value linear);

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order);

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape);

size_t linearize(ArrayRef<unsigned> multiDim, ArrayRef<unsigned> shape,
                 ArrayRef<unsigned> order);

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content);

inline bool isKernel(FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

inline Value getStackPointer(RewriterBase &rewriter,
                             FunctionOpInterface funcOp) {
  // See NOTE: [Additional Function Arguments]
  if (!isKernel(funcOp)) {
    return funcOp.getArgument(funcOp.getNumArguments() - 2);
  }

  auto mod = funcOp->getParentOfType<ModuleOp>();
  auto globalBase = dyn_cast<LLVM::GlobalOp>(mod.lookupSymbol("global_smem"));
  assert(globalBase);
  return rewriter.create<LLVM::AddressOfOp>(funcOp.getLoc(), globalBase);
}

inline Value getGlobalScratchPtr(Location loc, RewriterBase &rewriter,
                                 FunctionOpInterface funcOp,
                                 Value allocOffset = {}) {
  // See NOTE: [Additional Function Arguments]
  if (!isKernel(funcOp)) {
    // Base for this function
    auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() - 1);
    if (!allocOffset) {
      return gmemBase;
    }

    auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), 1);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    return b.gep(ptrTy, i8_ty, gmemBase, allocOffset);
  }

  // Base for entire kernel
  auto gmemBase = funcOp.getArgument(funcOp.getNumArguments() - 1);

  ModuleOp mod = funcOp.getOperation()->getParentOfType<ModuleOp>();
  auto allocSizeAttr = mod.getOperation()->getAttrOfType<mlir::IntegerAttr>(
      "ttg.global_scratch_memory_size");
  if (!allocSizeAttr) {
    return gmemBase;
  }

  Value gridIdx[3];
  Value gridDim[2];
  for (int k = 0; k < 3; ++k) {
    gridIdx[k] = rewriter.create<GetProgramIdOp>(loc, k);
  }
  for (int k = 0; k < 2; ++k) {
    gridDim[k] = rewriter.create<GetNumProgramsOp>(loc, k);
  }

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value linearId = gridIdx[2];
  for (int k = 0; k < 2; ++k) {
    linearId = b.add(gridIdx[1 - k], b.mul(linearId, gridDim[1 - k]));
  }

  auto allocSize = allocSizeAttr.getValue().getZExtValue();

  Value offset = b.mul(linearId, b.i32_val(allocSize));
  if (allocOffset) {
    offset = b.add(offset, allocOffset);
  }

  auto *ctx = rewriter.getContext();
  auto res =
      b.gep(mlir::LLVM::LLVMPointerType::get(ctx, 1), i8_ty, gmemBase, offset);
  return res;
}

inline Value getSharedMemoryBase(Location loc, RewriterBase &rewriter,
                                 const TargetInfoBase &target, Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                          target.getSharedAddressSpace());
  auto func = op->template getParentOfType<FunctionOpInterface>();
  if (!func)
    func = cast<FunctionOpInterface>(op);

  assert(op->hasAttr("allocation.offset"));
  size_t offset = cast<IntegerAttr>(op->getAttr("allocation.offset"))
                      .getValue()
                      .getZExtValue();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value offVal = b.i32_val(offset);
  Value base =
      b.gep(ptrTy, i8_ty, LLVM::getStackPointer(rewriter, func), offVal);
  return base;
}

// -----------------------------------------------------------------------
// MXFP utilities
// -----------------------------------------------------------------------

// Scale a mxfp4 value by a given scale.
Value mxfpScaleBf16(RewriterBase &rewriter, Location loc, Value v, Value scale,
                    bool fastMath);

} // namespace LLVM

// -----------------------------------------------------------------------
// Hardware Indices
// -----------------------------------------------------------------------

// If an operation is contained within a warp specialize region, this returns
// the thread ID offset of that warpgroup.
std::optional<int> getWarpGroupStartThreadId(Block *block);

// Returns CTA level thread ID.
Value getThreadId(OpBuilder &rewriter, Location loc);

// Get the lane ID, which is index of the thread within its warp.
Value getLaneId(OpBuilder &rewriter, Location loc);

// Get the lane ID and warp ID.
std::pair<Value, Value> getLaneAndWarpId(OpBuilder &rewriter, Location loc);

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
using LLVM::SharedMemoryObject;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::AMDMfmaEncodingAttr;
using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

inline Value dot(RewriterBase &rewriter, Location loc, ArrayRef<Value> offsets,
                 ArrayRef<Value> strides) {
  assert(offsets.size() == strides.size());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value ret = b.i32_val(0);
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    ret = b.add(ret, b.mul(offset, stride));
  }
  return ret;
}

/// Extend 2d shared object to 3d.
///
/// If tensor has 3 dimensions, returns original shared object.
/// If tensor shape is [M, N], return shared object describing shape [1, M, N]
///
/// This Function is used to simplify processing of 2d and 3d dot operands,
/// particularly in the conversion of local_load operation.
///
/// \param rewriter
/// \param loc
/// \param smemObj
/// \param shape shape of a tensor represented by smemObj
/// \returns shared object describing 3d tensor
SharedMemoryObject
getExpandedSharedMemoryObject(ConversionPatternRewriter &rewriter, Location loc,
                              SharedMemoryObject smemObj,
                              ArrayRef<int64_t> shape);

// "Applies" the given layout by computing layout(indices) and returning the
// resulting Values.
//
// In other words, this generates LLVM-dialect MLIR code to "run" the layout
// function.
SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(Location loc, RewriterBase &rewriter,
                  const LinearLayout &layout,
                  ArrayRef<std::pair<StringAttr, Value>> indices);

SmallVector<SmallVector<unsigned>> emitOffsetForLayout(Attribute layout,
                                                       RankedTensorType type);

// Emit indices calculation within each ConversionPattern, and returns a
// [elemsPerThread X rank] index matrix.
//
// For example, for a thread a owns `elemsPerThread` elements of a tensor with
// type `type` and layout `layout`, the result will contain `elemsPerThread`
// vectors. Each vector contains the SSA values of the indices required to
// access the corresponding element, starting from the inner dimension.
SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
            Attribute layout, RankedTensorType type, bool withCTAOffset);

// Emits IR to load data from shared memory into registers, or to store data
// from registers into shared memory.
//
// You supply perVectorCallback, which is called once per group of register
// elements to transfer.  You can use this callback to emit IR to load or store
// data from or to shared memory.
//
// elemLlvmTy should be dstTy's element type converted to an LLVM-dialect type.
//
// If maxVecElems is provided, we won't vectorize more than this many elements.
//
// Returns true on success.
[[nodiscard]] bool emitTransferBetweenRegistersAndShared(
    RankedTensorType registerTy, triton::gpu::MemDescType sharedTy,
    Type elemLlvmTy, std::optional<int32_t> maxVecElems,
    const SharedMemoryObject &smemObj, Location loc, RewriterBase &rewriter,
    const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback);

[[nodiscard]] bool emitTransferBetweenRegistersAndShared(
    LinearLayout &regLayout, triton::gpu::MemDescType sharedTy, Type elemLlvmTy,
    std::optional<int32_t> maxVecElems, const SharedMemoryObject &smemObj,
    Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback);

SmallVector<Value> loadSharedToDistributed(RankedTensorType dstTy,
                                           triton::gpu::MemDescType srcTy,
                                           Type elemLlvmTy,
                                           const SharedMemoryObject &smemObj,
                                           Location loc, RewriterBase &rewriter,
                                           const TargetInfoBase &target);

void storeDistributedToShared(
    triton::gpu::MemDescType dstTy, RankedTensorType srcTy, Type elemLlvmTy,
    ArrayRef<Value> srcVals, const SharedMemoryObject &smemObj, Location loc,
    RewriterBase &rewriter, const TargetInfoBase &target,
    std::pair<size_t, Type> *const llvmOpCount = nullptr);

inline SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
                                           RewriterBase &rewriter) {
  assert(bool(llvmStruct) && "can not unpack null values");
  if (llvmStruct.getType().isIntOrIndexOrFloat() ||
      isa<triton::PointerType>(llvmStruct.getType()) ||
      isa<LLVM::LLVMPointerType>(llvmStruct.getType()))
    return {llvmStruct};
  ArrayRef<Type> types =
      cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> results(types.size());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = b.extract_val(type, llvmStruct, i);
  }
  return results;
}

inline Value packLLElements(Location loc,
                            const LLVMTypeConverter *typeConverter,
                            ValueRange resultVals, RewriterBase &rewriter,
                            Type type) {
  auto structType =
      dyn_cast<LLVM::LLVMStructType>(typeConverter->convertType(type));
  if (!structType) {
    assert(resultVals.size() == 1);
    return *resultVals.begin();
  }

  auto elementTypes = structType.getBody();
  if (elementTypes.size() != resultVals.size()) {
    emitError(loc) << " size mismatch when packing elements for LLVM struct"
                   << " expected " << elementTypes.size() << " but got "
                   << resultVals.size();
  }
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  for (const auto &v : llvm::enumerate(resultVals)) {
    if (!v.value()) {
      emitError(loc)
          << "cannot insert null values into struct, but tried to insert"
          << v.value();
    }
    if (v.value().getType() != elementTypes[v.index()]) {
      LDBG("type " << type << " structType " << structType);
      LDBG("value " << v.value());
      emitError(loc) << "invalid element type in packLLElements. Expected "
                     << elementTypes[v.index()] << " but got "
                     << v.value().getType();
    }
    llvmStruct = b.insert_val(structType, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

inline SmallVector<Value> unpackLLVector(Location loc, Value llvmVec,
                                         RewriterBase &rewriter) {
  assert(bool(llvmVec) && "cannot unpack null value");
  if (llvmVec.getType().isIntOrIndexOrFloat() ||
      isa<triton::PointerType>(llvmVec.getType()) ||
      isa<LLVM::LLVMPointerType>(llvmVec.getType()))
    return {llvmVec};

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> results;
  for (int i = 0; i < cast<VectorType>(llvmVec.getType()).getNumElements();
       i++) {
    results.push_back(b.extract_element(llvmVec, b.i32_val(i)));
  }
  return results;
}

inline Value packLLVector(Location loc, ValueRange vals,
                          RewriterBase &rewriter) {
  assert(vals.size() > 0);
  auto vecType = vec_ty(vals[0].getType(), vals.size());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value vec = b.undef(vecType);
  for (int i = 0; i < vals.size(); i++) {
    vec = b.insert_element(vec, vals[i], b.i32_val(i));
  }
  return vec;
}

inline bool
isSimpleSharedMemoryAccess(ArrayRef<int64_t> shape,
                           ArrayRef<int64_t> allocShape,
                           triton::gpu::SharedEncodingTrait sharedEnc) {
  auto rank = shape.size();
  auto swizzledLayout =
      dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(sharedEnc);
  bool noSwizzling = swizzledLayout && swizzledLayout.getMaxPhase() == 1;
  return /*no swizzling*/ noSwizzling ||
         /*swizzling but same shape*/ shape == allocShape ||
         /*swizzling and rank-reduced and rank >= 2*/
         (shape == allocShape.take_back(rank) && rank >= 2);
}

inline llvm::MapVector<StringAttr, int32_t>
getAllFreeVarMasks(MLIRContext *ctx) {
  // Mask where all elements are redundant
  auto kReg = str_attr("reg");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  int32_t fullMask = -1;
  llvm::MapVector<StringAttr, int32_t> ret;
  for (auto dimName : {kReg, kLane, kWarp, kBlock}) {
    ret[dimName] = fullMask;
  }
  return ret;
}

inline llvm::MapVector<StringAttr, int32_t> getFreeVariableMasks(Type type) {
  auto ctx = type.getContext();
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  if (!tensorTy) {
    return getAllFreeVarMasks(ctx);
  }
  auto ll =
      triton::gpu::toLinearLayout(tensorTy.getShape(), tensorTy.getEncoding());
  return ll.getFreeVariableMasks();
}

inline bool isCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return (index & freeVarMask) == 0;
}

} // namespace mlir

#endif
