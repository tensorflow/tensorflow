/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace llvm {
class FastMathFlags;
class TargetOptions;
};

namespace xla {
namespace llvm_ir {

// Convert a absl::string_view to a llvm::StringRef. Note: both
// absl::string_view and llvm::StringRef are non-owning pointers into a
// string in memory. This method is used to feed strings to LLVM
// & Clang APIs that expect llvm::StringRef.
inline llvm::StringRef AsStringRef(absl::string_view str) {
  return llvm::StringRef(str.data(), str.size());
}

inline absl::string_view AsStringView(llvm::StringRef str) {
  return absl::string_view(str.data(), str.size());
}

template <typename T>
llvm::ArrayRef<T> AsArrayRef(const std::vector<T>& vec) {
  return llvm::ArrayRef<T>(vec.data(), vec.size());
}

template <typename T>
llvm::ArrayRef<T> AsArrayRef(const absl::Span<const T> slice) {
  return llvm::ArrayRef<T>(slice.data(), slice.size());
}

// Dump the given LLVM entity to a string. This works for Types and Values.
template <typename T>
string DumpToString(const T& entity) {
  std::string buffer_string;
  llvm::raw_string_ostream ostream(buffer_string);
  entity.print(ostream);
  ostream.flush();
  return buffer_string;
}

// Same as above, except that const T& does not work well with MILR because the
// print methods are not const.
template <typename T>
string DumpToString(T& entity) {
  std::string buffer_string;
  llvm::raw_string_ostream ostream(buffer_string);
  entity.print(ostream);
  ostream.flush();
  return buffer_string;
}

// Dump the given LLVM module to a string. This requires a function distinct
// from DumpToString because the signatures of the print() methods for Values
// and Modules are slightly different.
string DumpModuleToString(const llvm::Module& module);

// Constructs a human-friendly name from the given inputs.  The result is
// suitable for use as an llvm::Value's name.
//
// This is equivalent to
//
//   - changing the HloInstruction* to its name() (if we called that overload),
//   - joining all of the nonempty inputs by '.', and then
//   - removing all '%'s.
//
string IrName(absl::string_view a);
string IrName(absl::string_view a, absl::string_view b);
string IrName(const HloInstruction* a, absl::string_view b = "");

// Removes special characters from a function name.
//
// Note that this can cause different inputs to map to the same output, so after
// sanitizing a function name, you must run it through a uniquer.
string SanitizeFunctionName(string function_name);

// Emits a call to the specified intrinsic with the given operands. Overloaded
// intrinsics (for example, "minnum") must include a type in overloaded_types
// for each overloaded type. Typically, overloaded intrinsics have only a single
// overloaded type.
llvm::CallInst* EmitCallToIntrinsic(
    llvm::Intrinsic::ID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b,
    absl::string_view name = "");

// Emit float max. Emit maxnum intrinsic is fast math is disabled, or
// fcmp+select otherwise
llvm::Value* EmitFloatMax(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilder<>* b, bool enable_fast_min_max,
                          absl::string_view name = "");

// Emit float min. Emit minnum intrinsic is fast math is disabled, or
// fcmp+select otherwise
llvm::Value* EmitFloatMin(llvm::Value* lhs_value, llvm::Value* rhs_value,
                          llvm::IRBuilder<>* b, bool enable_fast_min_max,
                          absl::string_view name = "");

// Convenience methods for emitting a GEP instruction that indexes into a buffer
// (1-dimensional array), equivalent to array[index]. The type is automatically
// determined from the element type of the array.  The int64 index overload
// wraps the index in a i64 llvm::Value.
llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, llvm::Value* index,
                                   llvm::IRBuilder<>* b);
llvm::Value* EmitBufferIndexingGEP(llvm::Value* array, int64 index,
                                   llvm::IRBuilder<>* b);

// Returns the LLVM type which represents the given XLA primitive type.
llvm::Type* PrimitiveTypeToIrType(PrimitiveType element_type,
                                  llvm::Module* module);

// Returns the type size in bits. If "type" is a struct, it must be packed.
int GetSizeInBits(llvm::Type* type);

// Returns the LLVM type which represents the given XLA shape. For example,
// if "shape" is [5 x [10 x f32]], the function returns [5 x [10 x float]].
llvm::Type* ShapeToIrType(const Shape& shape, llvm::Module* module);

// Returns a value that represents a pointer to a global string constant that
// encodes the shape as a serialized protobuf.
StatusOr<llvm::Value*> EncodeSelfDescribingShapeConstant(const Shape& shape,
                                                         int32* shape_size,
                                                         llvm::IRBuilder<>* b);

// Converts a given literal to an IR Constant. Literals have known constant
// values at IR emission time.
llvm::Constant* ConvertLiteralToIrConstant(const Literal& literal,
                                           llvm::Module* module);

// Allocates a tile of shared memory.
llvm::GlobalVariable* AllocateSharedMemoryTile(llvm::Module* module,
                                               llvm::Type* tile_type,
                                               absl::string_view name);

// Inserts an allocate of the requested type at the entry point of the
// function that the builder is currently building. The insert point
// of the builder is set to the same place after calling this function
// as before.
//
// This can be useful to avoid e.g. executing an alloca every time
// through a loop.
llvm::AllocaInst* EmitAllocaAtFunctionEntry(llvm::Type* type,
                                            absl::string_view name,
                                            llvm::IRBuilder<>* b,
                                            int alignment = 0);

// As EmitAllocaAtFunctionEntry, but allocates element_count entries
// instead of a single element.
llvm::AllocaInst* EmitAllocaAtFunctionEntryWithCount(llvm::Type* type,
                                                     llvm::Value* element_count,
                                                     absl::string_view name,
                                                     llvm::IRBuilder<>* b,
                                                     int alignment = 0);

// Creates a basic block with the same context and function as for the
// builder. Inserts at the end of the function if insert_before is
// null.
llvm::BasicBlock* CreateBasicBlock(llvm::BasicBlock* insert_before,
                                   absl::string_view name,
                                   llvm::IRBuilder<>* b);

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
                          llvm::IRBuilder<>* b, bool emit_else = true);

// Emits a compare operation between "lhs" and "rhs" with the given predicate,
// and then converts the result to i8 so that it is addressable.
llvm::Value* EmitComparison(llvm::CmpInst::Predicate predicate,
                            llvm::Value* lhs, llvm::Value* rhs,
                            llvm::IRBuilder<>* b, absl::string_view name = "");

// Emits a call that logs the given value with the given tag as a prefix.
// The provided tag and value are passed to a runtime logging call that is
// embedded in this translation unit when the emitted code is executed.
//
// This can be very useful for debugging generated programs in short order when
// developing new generated routines.
//
// Precondition: value must be an int64.
// Precondition: tag must be a stable pointer for the lifetime of the generated
// program (the constant pointer is burned in to the program).
void EmitLogging(const char* tag, llvm::Value* value, llvm::IRBuilder<>* b);

// Adds alignment metadata to a load instruction using the given alignment.
// The alignment refers to the result of the load, not the load itself.
void SetAlignmentMetadataForLoad(llvm::LoadInst* load, uint64_t alignment);

// Adds dereferenceable metadata to a load instruction using the given
// the number of dereferenceable bytes.
// Dereferenceable refers to the result of the load, not the load itself.
void SetDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                       uint64_t dereferenceable_bytes);

// Tells LLVM `inst >= lower && inst < upper`. Returns `inst` for convenience.
llvm::Instruction* AddRangeMetadata(int64 lower, int64 upper,
                                    llvm::Instruction* inst);

void SetToFirstInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilder<>* builder);

void SetToLastInsertPoint(llvm::BasicBlock* blk, llvm::IRBuilder<>* builder);

// Create a bitwise rotation of `rotand` by `rotor`.
llvm::Value* CreateRor(llvm::Value* rotand, llvm::Value* rotor,
                       llvm::IRBuilder<>* builder);

// Returns the number of bytes within the shape.
int64 ByteSizeOf(const Shape& shape, const llvm::DataLayout& data_layout);

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

void DumpIrIfEnabled(mlir::ModuleOp mlir_module, int unique_id,
                     const DebugOptions& debug_options);

llvm::Function* CreateCpuFunction(llvm::FunctionType* function_type,
                                  llvm::GlobalValue::LinkageTypes linkage,
                                  const HloModuleConfig& module_config,
                                  absl::string_view name, llvm::Module* module);

// Extracts the xla_backend_extra_options from `config` and passes those that
// don't start with xla_ to LLVM.
void InitializeLLVMCommandLineOptions(const HloModuleConfig& config);

// Zero-extends two 32-bit values to 64 bits, multiplies them, and returns the
// result as a pair of (low 32 bits, high 32 bits).
std::pair<llvm::Value*, llvm::Value*> UMulLowHigh32(llvm::IRBuilder<>* b,
                                                    llvm::Value* src0,
                                                    llvm::Value* src1);
// Splits the 64-bit integer value into its high and low 32 bits.
std::pair<llvm::Value*, llvm::Value*> SplitInt64ToInt32s(
    llvm::IRBuilder<>* b, llvm::Value* value_64bits);

// Checks whether a global variable is already created to represent the state
// of a random number generator. If not, creates such a variable. Returns the
// global variable.
llvm::GlobalVariable* GetOrCreateVariableRngState(llvm::Module* module,
                                                  llvm::IRBuilder<>* b);

// Adds a delta value to the global state variable and return the old value of
// the variable.
llvm::Value* RngGetAndUpdateState(uint64 delta, llvm::Module* module,
                                  llvm::IRBuilder<>* b);

// Gets the LLVM address space that should be used for global variables (e.g.
// XLA's rng state).
unsigned GetGlobalMemoryAddressSpace(const llvm::Module& module);
}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_UTIL_H_
