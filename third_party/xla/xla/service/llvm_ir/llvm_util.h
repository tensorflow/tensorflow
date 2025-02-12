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

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
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

// We have different DumpToString functions for each type for findability. We
// use pointers / values based on the usual semantics of the parameter type.

std::string DumpToString(const llvm::Module* module);
std::string DumpToString(const llvm::Type* type);
std::string DumpToString(const llvm::Value* value);

// This also works for mlir::Op<...> descendants, such as mlir::ModuleOp.
//
// For findability:
//   std::string DumpToString(mlir::Op<...>& op);
//   std::string DumpToString(mlir::ModuleOp& module_op);
//
// The `operation` parameter is not const, because the used print() method is
// not const.
std::string DumpToString(mlir::Operation* operation);
std::string DumpToString(mlir::Type type);
std::string DumpToString(mlir::Value value);

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

}  // namespace llvm_ir
}  // namespace xla

#endif  // XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_
