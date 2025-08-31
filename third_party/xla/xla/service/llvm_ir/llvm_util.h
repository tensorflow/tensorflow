/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_
#define XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace llvm {
class FastMathFlags;
class TargetOptions;
};  // namespace llvm

namespace xla {
namespace llvm_ir {

// Overload for pointer types that allows to pass additional arguments to the
// print method.
template <typename T, typename... ARGS>
inline std::string DumpToString(T* entity, ARGS... args) {
  std::string s;
  llvm::raw_string_ostream ostream(s);
  entity->print(ostream, args...);
  return s;
}

// Overload for pointer types.
template <typename T>
inline std::string DumpToString(T* entity) {
  std::string s;
  llvm::raw_string_ostream ostream(s);
  ostream << *entity;
  return s;
}

// Overload for non-pointer types. Allows to pass additional arguments to the
// print method.
template <typename T, typename... ARGS>
inline std::string DumpToString(const T& entity, ARGS... args) {
  std::string s;
  llvm::raw_string_ostream ostream(s);
  entity.print(ostream, args...);
  return s;
}

template <typename T>
inline std::string DumpToString(const T& entity) {
  std::string s;
  llvm::raw_string_ostream ostream(s);
  ostream << entity;
  return s;
}

// Constructs a human-friendly name from the given inputs.  The result is
// suitable for use as an llvm::Value's name.
//
// This is equivalent to
//
//   - changing the HloInstruction* to its name() (if we called that overload),
//   - joining all of the nonempty inputs by '.', and then
//   - removing all '%'s.
//
std::string IrName(absl::string_view a);
std::string IrName(absl::string_view a, absl::string_view b);
std::string IrName(const HloInstruction* a, absl::string_view b = "");

// Construct a module from the given location with an optional name.
//
// The underlying "create" method is unsafe, because it leaks the new module by
// default. This function avoids this by always returning an OwningOpRef.
mlir::OwningOpRef<mlir::ModuleOp> CreateMlirModuleOp(
    mlir::Location loc, std::optional<llvm::StringRef> name = std::nullopt);

// Removes special characters from a function name.
//
// Note that this can cause different inputs to map to the same output, so after
// sanitizing a function name, you must run it through a uniquer.
std::string SanitizeFunctionName(std::string function_name);

// Emits a call to the specified intrinsic with the given operands. Overloaded
// intrinsics (for example, "minnum") must include a type in overloaded_types
// for each overloaded type. Typically, overloaded intrinsics have only a single
// overloaded type.
llvm::CallInst* EmitCallToIntrinsic(
    llvm::Intrinsic::ID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilderBase* b,
    absl::string_view name = "");

// Emit float max. Emit maxnum intrinsic is fast math is disabled, or
// fcmp+select otherwise
llvm::Value* EmitFloatMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name = "");

// Emit float min. Emit minnum intrinsic is fast math is disabled, or
// fcmp+select otherwise
llvm::Value* EmitFloatMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilderBase* b, bool enable_fast_min_max,
                          absl::string_view name = "");

// Convenience methods for emitting a GEP instruction that indexes into a buffer
// (1-dimensional array), equivalent to array[index]. The element type of the
// array must be explicitly passed in.  The int64_t index overload
// wraps the index in a i64 llvm::Value.
llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, llvm::Type* element_type,
                                   llvm::Value* index, llvm::IRBuilderBase* b);
llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, llvm::Type* element_type,
                                   int64_t index, llvm::IRBuilderBase* b);

// Returns the LLVM type which represents the given XLA primitive type.
llvm::Type* PrimitiveTypeToIrType(PrimitiveType element_type,
                                  llvm::LLVMContext& context);

// Returns the XLA primitive type which represents the given LLVM type.
// If `default_to_signed_for_integers` is true, then integer types will be
// treated as signed if they are not explicitly specified as unsigned.
PrimitiveType PrimitiveTypeFromIrType(
    llvm::Type* type, bool default_to_signed_for_integers = true);

// Returns the type size in bits. If "type" is a struct, it must be packed.
int GetSizeInBits(llvm::Type* type);

// Returns the LLVM type which represents the given XLA shape. For example,
// if "shape" is [5 x [10 x f32]], the function returns [5 x [10 x float]].
llvm::Type* ShapeToIrType(const Shape& shape, llvm::LLVMContext& context);

// Returns a value that represents a pointer to a global string constant that
// encodes the shape as a serialized protobuf.
absl::StatusOr<llvm::Value*> EncodeSelfDescribingShapeConstant(
    const Shape& shape, int32_t* shape_size, llvm::IRBuilderBase* b);

// Converts a given literal to an IR Constant. Literals have known constant
// values at IR emission time.
llvm::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           llvm::Module* module);

// Allocates a tile of shared memory.
llvm::GlobalVariable* AllocateSharedMemoryTile(llvm::Module* module,
                                               llvm::Type* tile_type,
                                               absl::string_view name);

// Utility class for working with shared memory.
class SharedMemoryTile {
 public:
  SharedMemoryTile() = default;
  explicit SharedMemoryTile(llvm::GlobalVariable* base_ptr,
                            llvm::Type* element_type)
      : base_ptr_(base_ptr), element_type_(element_type) {}

  llvm::Value* Address(absl::Span<llvm::Value* const> index,
                       llvm::IRBuilderBase* b) const;
  llvm::Value* Load(absl::Span<llvm::Value* const> index,
                    llvm::IRBuilderBase* b) const;
  llvm::StoreInst* Store(llvm::Value* value,
                         absl::Span<llvm::Value* const> index,
                         llvm::IRBuilderBase* b) const;
  llvm::Type* GetElementType() const { return element_type_; }

 private:
  llvm::GlobalVariable* base_ptr_;
  llvm::Type* element_type_;
};

SharedMemoryTile AllocateSharedMemoryTile(
    llvm::Module* module, llvm::Type* element_type,
    absl::Span<int64_t const> dimensions_major_to_minor,
    absl::string_view buffer_name);

// Inserts an allocate of the requested type at the entry point of the
// function that the builder is currently building. The insert point
// of the builder is set to the same place after calling this function
// as before.
//
// This can be useful to avoid e.g. executing an alloca every time
// through a loop.
llvm::AllocaInst* EmitAllocaAtFunctionEntry(llvm::Type* type,
                                            absl::string_view name,
                                            llvm::IRBuilderBase* b,
                                            int alignment = 0);

// As EmitAllocaAtFunctionEntry, but allocates element_count entries
// instead of a single element.
llvm::AllocaInst* EmitAllocaAtFunctionEntryWithCount(llvm::Type* type,
                                                     llvm::Value* element_count,
                                                     absl::string_view name,
                                                     llvm::IRBuilderBase* b,
                                                     int alignment = 0);

// Creates a basic block with the same context and function as for the
// builder. Inserts at the end of the function if insert_before is
// null.
llvm::BasicBlock* CreateBasicBlock(llvm::BasicBlock* insert_before,
                                   absl::string_view name,
                                   llvm::IRBuilderBase* b);

// Struct with data on a conditional branch in a diamond shape created
// via EmitIfThenElse.
struct LlvmIfData {
  // The block that has the conditional branch.
  llvm::BasicBlock* if_block;

  // The block that is executed if the condition is true.
  llvm::BasicBlock* true_block;

  // The block that is executed if the condition is false.
  llvm::BasicBlock* false_block;

  // The block that follows after both the true_block and the
  // false_block.
  llvm::BasicBlock* after_block;
};

// Inserts a diamond-shaped if-then-else construct at the current
// insertion point of the builder. This involves splitting the current
// block into two blocks, at the insertion point, and introducing a
// true-block and a false-block that connect the two split pieces. The
// true-block is executed if the condition parameter evaluates to true
// and otherwise the false-block is executed. If `emit_else` is false,
// it jumps to the after-block rather than the false-block if the
// condition is false, and the returned `false_block` is null.
//
// Currently the insertion point of the builder must be a well-formed
// block with a terminator. If you need to use this for a
// non-terminated block, just make the function able to do that too.
LlvmIfData EmitIfThenElse(llvm::Value* condition, absl::string_view name,
                          llvm::IRBuilderBase* b, bool emit_else = true);

// Emits a compare operation between "lhs" and "rhs" with the given predicate,
// and then converts the result to i8 so that it is addressable.
llvm::Value* EmitComparison(llvm::CmpInst::Predicate predicate,
                            llvm::Value* lhs, llvm::Value* rhs,
                            llvm::IRBuilderBase* b,
                            absl::string_view name = "");

// Emits a call that logs the given value with the given tag as a prefix.
// The provided tag and value are passed to a runtime logging call that is
// embedded in this translation unit when the emitted code is executed.
//
// This can be very useful for debugging generated programs in short order when
// developing new generated routines.
//
// Precondition: value must be an int64_t.
// Precondition: tag must be a stable pointer for the lifetime of the generated
// program (the constant pointer is burned in to the program).
void EmitLogging(const char* tag, llvm::Value* value, llvm::IRBuilderBase* b);

// Adds alignment metadata to a load instruction using the given alignment.
// The alignment refers to the result of the load, not the load itself.
void SetAlignmentMetadataForLoad(llvm::LoadInst* load, uint64_t alignment);

// Adds dereferenceable metadata to a load instruction using the given
// the number of dereferenceable bytes.
// Dereferenceable refers to the result of the load, not the load itself.
void SetDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                       uint64_t dereferenceable_bytes);

// Tells LLVM `inst >= lower && inst < upper`. Returns `inst` for convenience.
llvm::Instruction* AddRangeMetadata(int32_t lower, int32_t upper,
                                    llvm::Instruction* inst,
                                    llvm::Module* module);

void SetToFirstInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilderBase* builder);

void SetToLastInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilderBase* builder);

// Returns the number of bytes within the shape.
int64_t ByteSizeOf(const Shape& shape, const llvm::DataLayout& data_layout);

// Gets an llvm::FastMathFlags that reflects the settings in the given
// module config.
llvm::FastMathFlags GetCpuFastMathFlags(const HloModuleConfig& module_config);

// Computes a conservative union of the metadata in "a" and "b".  For
// aliasing-related metadata, this means the result can be applied to
// instructions whose aliasing relationship can be described either by "a" *or*
// by "b".
std::map<int, llvm::MDNode*> MergeMetadata(
    llvm::LLVMContext* context, const std::map<int, llvm::MDNode*>& a,
    const std::map<int, llvm::MDNode*>& b);

// Dumps out `llvm_module` to the path specified in DebugOptions, if dumping is
// enabled for the given HLO module.
//
// A sanitized version of `hlo_module_name` is incorporated into the file name.
// If `optimized` is true then a suffix of "-with-opt.ll" is used, else a suffix
// of "-no-opt.ll" is used.
void DumpIrIfEnabled(const HloModule& hlo_module,
                     const llvm::Module& llvm_module, bool optimized,
                     absl::string_view filename_suffix = "");

llvm::Function* CreateCpuFunction(llvm::FunctionType* function_type,
                                  llvm::GlobalValue::LinkageTypes linkage,
                                  const HloModuleConfig& module_config,
                                  absl::string_view name, llvm::Module* module);

// Checks whether a global variable is already created to represent the state
// of a random number generator. If not, creates such a variable. Returns the
// global variable.
llvm::GlobalVariable* GetOrCreateVariableRngState(llvm::Module* module,
                                                  llvm::IRBuilderBase* b);

// Adds a delta value to the global state variable and return the old value of
// the variable.
llvm::Value* RngGetAndUpdateState(uint64_t delta, llvm::Module* module,
                                  llvm::IRBuilderBase* b);

// Gets the LLVM address space that should be used for global variables (e.g.
// XLA's rng state).
unsigned GetGlobalMemoryAddressSpace();

// Emits a block which does "return void". Leaves the insert point as is.
llvm::BasicBlock* EmitReturnBlock(llvm::IRBuilderBase* b);

// Emits `if (condition) return`. Assumes that the current function returns
// void.
//
// Can either use a supplied `return_block`, or generate a new one.
void EmitEarlyReturn(llvm::Value* condition, llvm::IRBuilderBase* b,
                     llvm::BasicBlock* return_block = nullptr);

absl::StatusOr<llvm::Value*> EmitReducePrecisionIR(
    PrimitiveType src_ty, llvm::Value* x, int64_t dest_exponent_bits,
    int64_t dest_mantissa_bits, bool quiet_nans, llvm::IRBuilderBase* b);

template <PrimitiveType fx_type, int f8_exponent_bits>
llvm::Value* HandleHalfwayPointsFxToF8(llvm::Value* fx_abs_bits,
                                       llvm::Value* f8_bits,
                                       std::optional<size_t> vector_width,
                                       llvm::IRBuilderBase* b) {
  using llvm::APFloat;
  using llvm::APInt;
  using llvm::Value;
  static_assert(fx_type == F16 || fx_type == F32 || fx_type == F64);
  static_assert(3 <= f8_exponent_bits && f8_exponent_bits <= 4);

  const llvm::fltSemantics* fx_semantics;
  llvm::Type* ix_type;

  if constexpr (fx_type == F16) {
    fx_semantics = &llvm::APFloat::IEEEhalf();
    ix_type = b->getInt16Ty();
  } else if constexpr (fx_type == F32) {
    fx_semantics = &llvm::APFloat::IEEEsingle();
    ix_type = b->getInt32Ty();
  } else if constexpr (fx_type == F64) {
    fx_semantics = &llvm::APFloat::IEEEdouble();
    ix_type = b->getInt64Ty();
  }

  llvm::Type* i8_type = b->getInt8Ty();

  if (vector_width.has_value()) {
    ix_type = llvm::VectorType::get(
        ix_type, llvm::ElementCount::getFixed(*vector_width));
    i8_type = llvm::VectorType::get(
        i8_type, llvm::ElementCount::getFixed(*vector_width));
  }

  auto ix_const = [fx_semantics, ix_type](APFloat val) {
    bool losesInfo;
    val.convert(*fx_semantics, llvm::RoundingMode::NearestTiesToEven,
                &losesInfo);
    return llvm::ConstantInt::get(ix_type, val.bitcastToAPInt());
  };

  auto i8_const = [i8_type](int val) {
    return llvm::ConstantInt::get(i8_type, val);
  };

  // F16 values that are halfway between denormal F8 values. This is used to
  // determine how to round to denormal F8 values.
  const APFloat halfway_points_e4[8] = {
      APFloat(0x1.0p-10),  // halfway between [0/8 * 2^-6, 1/8 * 2^-6]
      APFloat(0x1.8p-9),   // halfway between [1/8 * 2^-6, 2/8 * 2^-6]
      APFloat(0x1.4p-8),   // halfway between [2/8 * 2^-6, 3/8 * 2^-6]
      APFloat(0x1.Cp-8),   // halfway between [3/8 * 2^-6, 4/8 * 2^-6]
      APFloat(0x1.2p-7),   // halfway between [4/8 * 2^-6, 5/8 * 2^-6]
      APFloat(0x1.6p-7),   // halfway between [5/8 * 2^-6, 6/8 * 2^-6]
      APFloat(0x1.Ap-7),   // halfway between [6/8 * 2^-6, 7/8 * 2^-6]
      APFloat(0x1.Ep-7)    // halfway between [7/8 * 2^-6, 8/8 * 2^-6]
  };

  const APFloat halfway_points_e3[16] = {
      APFloat(0x1.0p-7),  // halfway between [0/16 * 2^-2, 1/16 * 2^-2]
      APFloat(0x1.8p-6),  // halfway between [1/16 * 2^-2, 2/16 * 2^-2]
      APFloat(0x1.4p-5),  // halfway between [2/16 * 2^-2, 3/16 * 2^-2]
      APFloat(0x1.Cp-5),  // halfway between [3/16 * 2^-2, 4/16 * 2^-2]
      APFloat(0x1.2p-4),  // halfway between [4/16 * 2^-2, 5/16 * 2^-2]
      APFloat(0x1.6p-4),  // halfway between [5/16 * 2^-2, 6/16 * 2^-2]
      APFloat(0x1.Ap-4),  // halfway between [6/16 * 2^-2, 7/16 * 2^-2]
      APFloat(0x1.Ep-4),  // halfway between [7/16 * 2^-2, 8/16 * 2^-2]
      APFloat(0x1.1p-3),  // halfway between [8/16 * 2^-2, 9/16 * 2^-2]
      APFloat(0x1.3p-3),  // halfway between [9/16 * 2^-2, 10/16 * 2^-2]
      APFloat(0x1.5p-3),  // halfway between [10/16 * 2^-2, 11/16 * 2^-2]
      APFloat(0x1.7p-3),  // halfway between [11/16 * 2^-2, 12/16 * 2^-2]
      APFloat(0x1.9p-3),  // halfway between [12/16 * 2^-2, 13/16 * 2^-2]
      APFloat(0x1.Bp-3),  // halfway between [13/16 * 2^-2, 14/16 * 2^-2]
      APFloat(0x1.Dp-3),  // halfway between [14/16 * 2^-2, 15/16 * 2^-2]
      APFloat(0x1.Fp-3),  // halfway between [15/16 * 2^-2, 16/16 * 2^-2]
  };

  const APFloat* halfway_points;
  int arr_sz;
  if constexpr (f8_exponent_bits == 4) {
    halfway_points = halfway_points_e4;
    arr_sz = 8;
  } else if constexpr (f8_exponent_bits == 3) {
    halfway_points = halfway_points_e3;
    arr_sz = 16;
  }

  // Handle case where output is denormal. If we're rounding to a denormal
  // value, ignore the current value of f8_bits and set it to the correct
  // denormal value. We emit the equivalent of the following:
  //
  //   if (f16_abs_bits <= halfway_points[0]) {
  //     f8_bits = 0;
  //   } else if (f16_abs_bits < halfway_points[1]) {
  //     f8_bits = 1;
  //   } else if (f16_abs_bits <= halfway_points[2]) {
  //   ...  // More if-else statements. The comparisons alternate between <=
  //   ...  // and < to handle round-to-even properly.
  //   } else if (f16_abs_bits < halfway_points[7])  {
  //     f8_bits = 7;
  //   }
  for (int i = arr_sz - 1; i >= 0; i--) {
    Value* comparison;
    llvm::Constant* half_way_point = ix_const(halfway_points[i]);

    if (i % 2 == 0) {
      comparison = b->CreateICmpULE(fx_abs_bits, half_way_point);
    } else {
      comparison = b->CreateICmpULT(fx_abs_bits, half_way_point);
    }

    f8_bits = b->CreateSelect(comparison, i8_const(i), f8_bits);
  }

  return f8_bits;
}

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_
