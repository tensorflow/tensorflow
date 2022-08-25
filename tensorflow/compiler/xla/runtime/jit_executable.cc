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

#include "tensorflow/compiler/xla/runtime/jit_executable.h"

#include <memory>
#include <string_view>
#include <utility>

#include "llvm/ADT/Optional.h"
#include "tensorflow/compiler/xla/mlir/utils/runtime/constraints.h"
#include "tensorflow/compiler/xla/runtime/errors.h"

namespace xla {
namespace runtime {

using absl::StatusOr;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

using llvm::ArrayRef;
using llvm::ErrorOr;
using llvm::Expected;
using llvm::Optional;

using tfrt::MakeAvailableAsyncValueRef;
using tfrt::MakeErrorAsyncValueRef;

using Specialization = JitExecutable::Specialization;

static bool IsSpecializationOnly(ArrayRef<ArgumentConstraint> constraints) {
  return llvm::any_of(constraints, [](ArgumentConstraint constraint) {
    return constraint != ArgumentConstraint::kResolved;
  });
}

static bool HasValueConstraints(ArrayRef<ArgumentConstraint> constraints) {
  return llvm::any_of(constraints, [](ArgumentConstraint constraint) {
    return constraint == ArgumentConstraint::kValue;
  });
}

// Returns true if all function operands have statically known shape.
static bool HasStaticShapeOperands(const FunctionType& signature) {
  auto is_dynamic = [](absl::Span<const int64_t> sizes) -> bool {
    return llvm::any_of(sizes, mlir::ShapedType::isDynamic);
  };

  for (unsigned i = 0; i < signature.num_operands(); ++i) {
    const Type* type = signature.operand(i);

    // Get the underlying value type from the async value.
    while (auto* value = dyn_cast<AsyncValueType>(type))
      type = &value->value_type();

    // Unranked types do not have statically known shape.
    if (isa<UnrankedTensorType, UnrankedMemrefType>(type)) return false;

    // For ranked memrefs and tensors check known sizes.
    if (auto* memref = dyn_cast<MemrefType>(type))
      if (is_dynamic(memref->sizes())) return false;
    if (auto* tensor = dyn_cast<RankedTensorType>(type))
      if (is_dynamic(tensor->sizes())) return false;

    // All other types are non-shaped and thus have "statically known shape".

    // TODO(ezhulenev): Run time types might need to support type interfaces or
    // a hierarchy with a base `ShapedType` so that users can define their own
    // types that can participate in shape specialization. This becomes
    // complicated for container-like types (e.g. async value) that might
    // contain a nested type that is shaped (e.g. memref). For now only the
    // canonical types can participate in shape specialization.
  }

  return true;
}

/*static*/ void JitExecutable::InlineCompilationTaskRunner(
    size_t num_specializations, ArrayRef<ArgumentConstraint> constraints,
    ArgumentsRef arguments, CompilationTask task, UserData user_data) {
  task();
}

/*static*/ Expected<JitExecutable> JitExecutable::Instantiate(
    std::string_view mlir_module, std::string_view entrypoint, Options opts,
    std::string_view memory_region_name, CompilationTaskRunner runner) {
  // Try to instantiate compilation context from the mlir source.
  Expected<std::unique_ptr<JitCompiler>> compiler =
      JitCompiler::Instantiate(opts.compiler, mlir_module, entrypoint);
  if (auto err = compiler.takeError()) return std::move(err);

  // Get resolved operands constraints for the entrypoint function.
  auto constraints = GetArgumentsConstraints((*compiler)->entrypoint());
  if (auto err = constraints.takeError()) return std::move(err);

  // Get the entrypoint function signature, it will be later required to
  // compute the specialized function signature from the operands at runtime.
  auto signature = opts.compiler.type_converter.Convert(
      (*compiler)->entrypoint().getFunctionType());
  if (auto err = signature.takeError()) return std::move(err);

  // If all of the operands have static shape, then we can always use default
  // binary for execution (unless specialization is explicitly required by the
  // operands constraints).
  if (HasStaticShapeOperands(*signature) && !IsSpecializationOnly(*constraints))
    opts.specialization = Specialization::kDisabled;

  // Return an error if specialization is explicitly disabled, yet some of
  // the operands have unresolved constraints.
  if (opts.specialization == Specialization::kDisabled &&
      IsSpecializationOnly(*constraints))
    return MakeStringError(
        "compilation options disabled specialization, yet operands have "
        "unresolved constraints: ",
        *constraints);

  // If the module must be specialized, return JitExecutable without a default
  // compiled executable.
  if (opts.specialization == Specialization::kAlways ||
      IsSpecializationOnly(*constraints))
    return JitExecutable(mlir_module, entrypoint, memory_region_name,
                         std::move(opts), std::move(*constraints),
                         std::move(*signature),
                         /*default_executable=*/llvm::None, std::move(runner));

  // Otherwise try to compile the default executable.
  Expected<Executable> executable =
      JitCompiler::Compile(std::move(*compiler), memory_region_name);
  if (auto err = executable.takeError()) return std::move(err);

  return JitExecutable(mlir_module, entrypoint, memory_region_name,
                       std::move(opts), std::move(*constraints),
                       std::move(*signature), std::move(*executable),
                       std::move(runner));
}

JitExecutable::JitExecutable(std::string_view mlir_module,
                             std::string_view entrypoint,
                             std::string_view memory_region_name, Options opts,
                             ArrayRef<ArgumentConstraint> constraints,
                             FunctionType signature,
                             Optional<Executable> default_executable,
                             CompilationTaskRunner runner)
    : mlir_module_(mlir_module),
      entrypoint_(entrypoint),
      memory_region_name_(memory_region_name),
      opts_(std::move(opts)),
      constraints_(constraints.begin(), constraints.end()),
      has_value_constraints_(HasValueConstraints(constraints_)),
      signature_(std::move(signature)),
      symbolic_shapes_resolver_(signature_, constraints_),
      has_default_executable_(default_executable.has_value()),
      runner_(std::move(runner)),
      specializations_(std::make_unique<Specializations>()) {
  // Initialize default executable if it is available.
  if (has_default_executable_) {
    default_executable_ =
        MakeAvailableAsyncValueRef<Executable>(std::move(*default_executable));
  } else {
    default_executable_ =
        MakeErrorAsyncValueRef("default executable is not available");
  }
}

AsyncValuePtr<Executable> JitExecutable::DefaultExecutable() const {
  return default_executable_.AsPtr();
}

ArrayRef<ArgumentConstraint> JitExecutable::constraints() const {
  return constraints_;
}

// Combines `hash` with a hash value computed from a value constrained operands.
static llvm::hash_code CombineWithValueConstraineOperands(
    llvm::hash_code hash, ArgumentsRef arguments,
    ArrayRef<ArgumentConstraint> constraints) {
  for (int i = 0; i < constraints.size(); ++i) {
    if (LLVM_LIKELY(constraints[i] != ArgumentConstraint::kValue)) continue;

    // TODO(ezhulenev): Currently we only support value specialization of Tensor
    // operands (with MemrefDesc run time argument), it should be extended to
    // support open type and argument hierarchies.
    const MemrefDesc& memref = cast<MemrefDesc>(arguments[i]);
    const auto* data = static_cast<uint8_t*>(memref.data());
    size_t rank = memref.rank();
    assert(rank == 0 || rank == 1);
    size_t num_values = rank == 0 ? 1 : memref.size(0);
    int64_t len = num_values * primitive_util::ByteWidth(memref.dtype());
    hash = llvm::hash_combine(hash, llvm::hash_combine_range(data, data + len));
  }
  return hash;
}

// TODO(ezhulenev): The fast path should be free of mutex to find the
// pre-compiled specialization. Maybe use atomic pointers (multiple atomic
// pointers?) to keep the most commonly used specialization available without
// doing a lookup in the AsyncValuesCache.
//
// TODO(ezhulenev): The number of specializations should be bounded, ideally we
// should only keep N most common specializations, and for everything else
// fall back on the default executable. However what to do if default executable
// is not available, and the number of specializations is above N?
Expected<AsyncValuePtr<Executable>> JitExecutable::GetExecutable(
    ArgumentsRef arguments, UserData user_data,
    const SpecializationListener* listener) {
  // Do not try to compile specialized executable if it is explicitly disabled.
  if (opts_.specialization == Specialization::kDisabled)
    return DefaultExecutable();

  // The number of arguments must match the entrypoint signature.
  if (LLVM_UNLIKELY(arguments.size() != signature_.num_operands()))
    return MakeStringError("expected ", signature_.num_operands(),
                           " arguments, got: ", arguments.size());

  // Resolve symbolic shapes hash based on the static and runtime information.
  //
  // We rely on the hash code to find the specialized executable. In case of
  // a collision (practically impossible) incompatible arguments will be
  // rejected by the executable arguments verification.
  StatusOr<llvm::hash_code> hash =
      symbolic_shapes_resolver_.ResolveHash(arguments);

  // If we failed to resolve the symbolic shapes hash, then we need to verify
  // all the operands to find the mismatch and report it to the user.
  if (LLVM_UNLIKELY(!hash.ok())) {
    for (unsigned i = 0; i < arguments.size(); ++i) {
      auto* type = signature_.operand(i);

      // TODO(ezhulenev): Support open shaped type/argument hierarchy.
      auto* memref_arg = dyn_cast<MemrefDesc>(&arguments[i]);
      if (!memref_arg) continue;

      if (auto* memref = dyn_cast<MemrefType>(type)) {
        if (auto err = VerifyMemrefArgument(i, *memref, *memref_arg))
          return std::move(err);

      } else if (auto* tensor = dyn_cast<RankedTensorType>(type)) {
        if (auto err = VerifyMemrefArgument(i, *tensor, *memref_arg))
          return std::move(err);

      } else {
        return MakeStringError("expected shaped operand at #", i,
                               ", got: ", *signature_.operand(i));
      }
    }

    assert(false && "failed to detect incorrect operand");
    return MakeStringError("failed to resolve symbolic shapes");
  }

  // Combine with a hash value computed from the value constrained operands.
  if (LLVM_UNLIKELY(has_value_constraints_))
    *hash = CombineWithValueConstraineOperands(*hash, arguments, constraints_);

  // Maybe return Executable from the cache.
  if (auto cached = specializations_->Find(*hash)) {
    // Always use specialized kernel if required by the compilation options.
    if (opts_.specialization == Specialization::kAlways) return cached;

    // Fall back on default executable if the specialization is not yet
    // available.
    if (has_default_executable_ && !cached.IsAvailable())
      return DefaultExecutable();

    return cached;
  }

  // Instantiation from the source and specialization are cheap, so we do it in
  // the caller thread. We only use compilation runner for expensive part.

  // Try to instantiate compilation context from the mlir source.
  Expected<std::unique_ptr<JitCompiler>> compiler =
      JitCompiler::Instantiate(opts_.compiler, mlir_module_, entrypoint_);

  if (auto err = compiler.takeError()) {
    assert(false && "parsing mlir module must always succeed at this point");
    return std::move(err);
  }

  // Specialize executable to the concrete operands.
  StatusOr<llvm::SmallVector<SymbolicShapesResolver::SymbolicShape>>
      symbolic_shapes = symbolic_shapes_resolver_.Resolve(arguments);
  if (auto err = (*compiler)->Specialize(arguments, *symbolic_shapes,
                                         constraints_, listener)) {
    return MakeStringError("failed to specialize executable: ", err);
  }

  // Allocate a placeholder for the compiled specialization only after we are
  // ready to dispatch the compilation task.
  Specializations::Entry entry = specializations_->Allocate(*hash);

  // We lost the race; some other invocation will do the compilation.
  if (!entry.allocated) return entry.ptr;

  // Get the specialization id from the size of the specializations cache.
  size_t specialization = entry.size - 1;

  // Construct the task that will do the specialized executable compilation.
  auto compile = CompilationTask(
      [compiler = std::move(*compiler), ref = entry.ptr.CopyRef(),
       memory_region_name = memory_region_name_, specialization]() mutable {
        Expected<Executable> executable = JitCompiler::Compile(
            std::move(compiler), memory_region_name, specialization);

        // Set the allocated entry async value state to error or concrete.
        if (auto err = executable.takeError()) {
          ref.SetError(std::move(err));
        } else {
          ref.emplace(std::move(*executable));
        }
      });

  // Offload specialization compilation to the user provided runner.
  runner_(specialization, constraints_, arguments, std::move(compile),
          user_data);

  // Use the default executable while we are compiling a specialized version if
  // this is not explicitly disabled by the compilation options.
  if (opts_.specialization == Specialization::kAlways)
    return entry.ptr;
  else
    return has_default_executable_ ? DefaultExecutable() : entry.ptr;
}

AsyncValueRef<Chain> JitExecutable::AllExecutablesCompiled() const {
  return specializations_->AllAvailable();
}

}  // namespace runtime
}  // namespace xla
