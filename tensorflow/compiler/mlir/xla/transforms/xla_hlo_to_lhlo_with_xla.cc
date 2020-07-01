/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/transforms/xla_hlo_to_lhlo_with_xla.h"

#include <memory>
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

using xla::BufferAllocation;
using xla::BufferAssignment;
using xla::HloComputation;
using xla::HloInstruction;
using xla::HloModule;
using xla::HloModuleProto;
using xla::HloProto;
using xla::Shape;
using xla::StatusOr;

namespace mlir {
namespace {

absl::string_view StringRefToView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}

StatusOr<std::unique_ptr<HloModule>> HloModuleFromProto(
    const HloProto& hlo_proto) {
  const HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(const ::xla::HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          module_proto, ::xla::GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(module_proto, module_config);
}

// This class will process an HloModule with the supplied BufferAssignment and
// populate the MLIR ModuleOp with the computation converted in the LHLO
// dialect.
class LhloDialectEmitter : public ::xla::DfsHloVisitorWithDefault {
 public:
  // Main entry point of the processing: after this call the MLIR ModuleOp is
  // populated with the computation from the HloModule. The returned `Status`
  // indicates success or failure in the conversion.
  Status Run();

  LhloDialectEmitter(const BufferAssignment& assignment,
                     const HloModule& hlo_module, ModuleOp module)
      : assignment_(std::move(assignment)),
        hlo_module_(hlo_module),
        module_(module),
        builder_(module.getContext()),
        i8_type_(builder_.getIntegerType(8)) {}

 private:
  template <typename OpType>
  StatusOr<OpType> CreateOpWithoutAttrs(HloInstruction* instr);

  Status DefaultAction(HloInstruction* instr) final;

  // Computation parameters don't need any specific handling when they are
  // visited, they are already processed when we enter a new computation.
  Status HandleParameter(HloInstruction* instr) final { return Status::OK(); }

  Status HandleSort(HloInstruction* instr) final;

  // Helper function that recursively visits the tuple structure in
  // `current_shape`, and reconstruct a matching xla_lhlo::TupleOp.
  // Each leaf node is converted to an std.view op with corresponding offsets.
  // If no tuple presents, it simply returns a view of the buffer.
  Status CreateView(const HloInstruction* instr, const Shape& current_shape,
                    ::xla::ShapeIndex* current_shape_index,
                    SmallVectorImpl<Value>* values);

  // Helper function to create view/tuple of views to a buffer for a given
  // instruction result.
  Status GetOrCreateView(const HloInstruction* instr,
                         SmallVectorImpl<Value>* values);

  // Return an MLIR location for an HLO instruction.
  Location getLocation(HloInstruction* inst) {
    return NameLoc::get(builder_.getIdentifier(inst->name()),
                        builder_.getContext());
  }

  // This map provides access to MLIR buffers for each HLO buffer allocation.
  // The MLIR buffers are all `memref<{size}xi8>` and correspond to function
  // parameters. It is populated at the beginning of the processing with all the
  // buffer allocations and is unchanged afterward. Every HLOInstruction is
  // using a "slice" of the buffer allocation and providing shape, layout, and
  // Dtype. An MLIR view is used separately to model slices into the allocations
  // (see below).
  llvm::DenseMap<const BufferAllocation*, Value> allocations_;

  // This map provides access to MLIR buffers for each HLO instruction, keyed by
  // its buffer slice. A slice is contained in a BufferAllocation, and has an
  // offset and a size.
  //
  // As for why we don't use HloInstruction*, see GetOrCreateView(), but mostly
  // we want to leverage better of the aliased buffers.
  //
  // If the HloInstruction is a tuple, all leaf nodes are stored flattened.
  // Otherwise, there will be a single buffer.
  //
  // An MLIR buffer is either an input parameter, or a ViewOp in the case where
  // the slice is only part of its allocation.
  //
  // `slices_` is populated lazily in the `GetOrCreateView()` helper as we
  // process every instruction.
  using SliceKey = std::tuple<const BufferAllocation*, int64_t, int64_t>;
  llvm::DenseMap<SliceKey, llvm::SmallVector<Value, 1>> slices_;

  // The BufferAssignment computed by XLA ahead of time.
  const BufferAssignment& assignment_;

  // The HLO module that will be converted.
  const HloModule& hlo_module_;

  // This is the MLIR module in which a function will be created for every HLO
  // computation.
  ModuleOp module_;

  // The builder keeps track of the current insertion point in the MLIR module.
  OpBuilder builder_;
  // Convenient "cached" access to this widely used MLIR type (i8).
  Type i8_type_;
};

template <typename OpType>
StatusOr<OpType> LhloDialectEmitter::CreateOpWithoutAttrs(
    HloInstruction* instr) {
  Location loc = getLocation(instr);
  ArrayRef<std::pair<Identifier, Attribute>> attrs;
  ArrayRef<Type> rets{};

  llvm::SmallVector<Value, 4> operands;
  for (const HloInstruction* operand : instr->operands()) {
    TF_RETURN_IF_ERROR(GetOrCreateView(operand, &operands));
  }
  TF_RETURN_IF_ERROR(GetOrCreateView(instr, &operands));

  return builder_.create<OpType>(loc, rets, operands, attrs);
}

Status LhloDialectEmitter::DefaultAction(HloInstruction* instr) {
  using ::xla::HloOpcode;
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
      return CreateOpWithoutAttrs<xla_lhlo::AbsOp>(instr).status();
    case HloOpcode::kAdd:
      return CreateOpWithoutAttrs<xla_lhlo::AddOp>(instr).status();
    case HloOpcode::kAnd:
      return CreateOpWithoutAttrs<xla_lhlo::AndOp>(instr).status();
    case HloOpcode::kCeil:
      return CreateOpWithoutAttrs<xla_lhlo::CeilOp>(instr).status();
    case HloOpcode::kComplex:
      return CreateOpWithoutAttrs<xla_lhlo::ComplexOp>(instr).status();
    case HloOpcode::kCopy:
      return CreateOpWithoutAttrs<xla_lhlo::CopyOp>(instr).status();
    case HloOpcode::kCos:
      return CreateOpWithoutAttrs<xla_lhlo::CosOp>(instr).status();
    case HloOpcode::kDivide:
      return CreateOpWithoutAttrs<xla_lhlo::DivOp>(instr).status();
    case HloOpcode::kExp:
      return CreateOpWithoutAttrs<xla_lhlo::ExpOp>(instr).status();
    case HloOpcode::kImag:
      return CreateOpWithoutAttrs<xla_lhlo::ImagOp>(instr).status();
    case HloOpcode::kLog:
      return CreateOpWithoutAttrs<xla_lhlo::LogOp>(instr).status();
    case HloOpcode::kMaximum:
      return CreateOpWithoutAttrs<xla_lhlo::MaxOp>(instr).status();
    case HloOpcode::kMinimum:
      return CreateOpWithoutAttrs<xla_lhlo::MinOp>(instr).status();
    case HloOpcode::kMultiply:
      return CreateOpWithoutAttrs<xla_lhlo::MulOp>(instr).status();
    case HloOpcode::kNegate:
      return CreateOpWithoutAttrs<xla_lhlo::NegOp>(instr).status();
    case HloOpcode::kReal:
      return CreateOpWithoutAttrs<xla_lhlo::RealOp>(instr).status();
    case HloOpcode::kRemainder:
      return CreateOpWithoutAttrs<xla_lhlo::RemOp>(instr).status();
    case HloOpcode::kRsqrt:
      return CreateOpWithoutAttrs<xla_lhlo::RsqrtOp>(instr).status();
    case HloOpcode::kSelect:
      return CreateOpWithoutAttrs<xla_lhlo::SelectOp>(instr).status();
    case HloOpcode::kSign:
      return CreateOpWithoutAttrs<xla_lhlo::SignOp>(instr).status();
    case HloOpcode::kSqrt:
      return CreateOpWithoutAttrs<xla_lhlo::SqrtOp>(instr).status();
    case HloOpcode::kSubtract:
      return CreateOpWithoutAttrs<xla_lhlo::SubOp>(instr).status();
    case HloOpcode::kTanh:
      return CreateOpWithoutAttrs<xla_lhlo::TanhOp>(instr).status();
    default:
      llvm::errs() << instr->ToString();
      return tensorflow::errors::Internal(
          absl::StrCat("LHLO opcode ", ::xla::HloOpcodeString(instr->opcode()),
                       " is not supported."));
  }
  return Status::OK();
}

Status LhloDialectEmitter::HandleSort(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto sort, CreateOpWithoutAttrs<xla_lhlo::SortOp>(instr));
  auto* sort_instr = ::xla::Cast<::xla::HloSortInstruction>(instr);
  sort.dimensionAttr(builder_.getI64IntegerAttr(sort_instr->sort_dimension()));
  sort.is_stableAttr(builder_.getBoolAttr(sort_instr->is_stable()));
  TF_RETURN_IF_ERROR(::xla::HloFunctionImporter::ImportAsRegion(
      *sort_instr->called_computations()[0], &sort.comparator(), &builder_));
  return Status::OK();
}

Status LhloDialectEmitter::CreateView(const HloInstruction* instr,
                                      const Shape& current_shape,
                                      ::xla::ShapeIndex* current_shape_index,
                                      SmallVectorImpl<Value>* values) {
  if (current_shape.IsTuple()) {
    for (int i = 0; i < current_shape.tuple_shapes().size(); i++) {
      current_shape_index->push_back(i);
      TF_RETURN_IF_ERROR(CreateView(instr, current_shape.tuple_shapes(i),
                                    current_shape_index, values));
      current_shape_index->pop_back();
    }
    return Status::OK();
  }

  TF_ASSIGN_OR_RETURN(Type out_type, ::xla::ConvertShapeToType<MemRefType>(
                                         current_shape, builder_));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      assignment_.GetUniqueSlice(instr, *current_shape_index));
  Value alloc = allocations_[slice.allocation()];
  if (alloc.getType() == out_type) {
    values->push_back(alloc);
    return Status::OK();
  }

  Value byte_shift =
      builder_.create<ConstantIndexOp>(alloc.getLoc(), slice.offset());
  values->push_back(builder_.create<ViewOp>(builder_.getUnknownLoc(), out_type,
                                            alloc, byte_shift,
                                            /*sizes=*/ValueRange{}));
  return Status::OK();
}

// Returns a view for the result of an instruction.
// We first get a view for the slice in the allocation, and then may need to
// create another view to adjust the slice for the shape of the instruction.
Status LhloDialectEmitter::GetOrCreateView(const HloInstruction* instr,
                                           SmallVectorImpl<Value>* values) {
  // In terms of cache key, we have several choices:
  // * Use `instr`. It's the easiest, but it creates different cache entries for
  // aliased buffers, which could have been deduplicated.
  // * Use the actual content as the key, aka a tree of allocation slices.
  // * Somewhere in the middle, use the allocation slice for the instruction. If
  // `instr` is a tuple, the key is the allocated buffer for the tuple itself
  // (an array of pointers).
  //
  // We choose the third approach for simplicity.
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      assignment_.GetUniqueTopLevelSlice(instr));
  SliceKey slice_key(slice.allocation(), slice.offset(), slice.size());
  auto result = slices_.try_emplace(slice_key, llvm::SmallVector<Value, 1>{});
  llvm::SmallVectorImpl<Value>& new_values = result.first->second;
  if (result.second) {
    ::xla::ShapeIndex shape_index;
    TF_RETURN_IF_ERROR(
        CreateView(instr, instr->shape(), &shape_index, &new_values));
  }
  values->insert(values->end(), new_values.begin(), new_values.end());
  return Status::OK();
}

Status LhloDialectEmitter::Run() {
  HloComputation* computation = hlo_module_.entry_computation();
  std::string function_name =
      computation->name().empty() ? "__compute" : computation->name();

  // Create the function as () -> (), we'll compute the arguments from the
  // buffer allocation and update the type then.
  auto func_op = FuncOp::create(builder_.getUnknownLoc(), function_name,
                                builder_.getFunctionType({}, {}));
  Block* block = func_op.addEntryBlock();

  llvm::SmallVector<const BufferAllocation*, 8> ordered_allocations;
  for (const BufferAllocation& alloc : assignment_.Allocations())
    ordered_allocations.push_back(&alloc);

  // Sort the rather arbitrarily ordered allocations to match the input/output
  // parameters. Specifically We want to sort buffer allocations in the
  // following order:
  // * Parameters always order before non-parameters.
  // * Different parameters order by parameter number.
  // * Different allocations for the same parameter order by the shape index.
  //
  // TODO(timshen): there should be only one non-parameter buffer, the temp
  // buffer. Check on that.
  const auto allocation_comparator = [](const BufferAllocation* lhs,
                                        const BufferAllocation* rhs) {
    if (lhs->is_entry_computation_parameter() !=
        rhs->is_entry_computation_parameter()) {
      return lhs->is_entry_computation_parameter() >
             rhs->is_entry_computation_parameter();
    }
    if (lhs->is_entry_computation_parameter()) {
      return std::tuple<int, const ::xla::ShapeIndex&>(
                 lhs->parameter_number(), lhs->param_shape_index()) <
             std::tuple<int, const ::xla::ShapeIndex&>(
                 rhs->parameter_number(), rhs->param_shape_index());
    }
    return false;
  };

  std::stable_sort(ordered_allocations.begin(), ordered_allocations.end(),
                   allocation_comparator);

  // The function signature will be composed of:
  // - one memref for each of the parameters.
  // - one memref for each other buffer allocation.
  llvm::SmallVector<MutableDictionaryAttr, 8> args_attrs;
  for (const BufferAllocation* alloc : ordered_allocations) {
    if (alloc->is_entry_computation_parameter()) {
      const ::xla::Shape& buffer_shape = ::xla::ShapeUtil::GetSubshape(
          computation->parameter_instruction(alloc->parameter_number())
              ->shape(),
          alloc->param_shape_index());

      TF_ASSIGN_OR_RETURN(auto arg_type, ::xla::ConvertShapeToType<MemRefType>(
                                             buffer_shape, builder_));

      // First map parameters to memrefs on the operation.
      block->addArgument(arg_type);
      allocations_[alloc] = block->getArguments().back();
      args_attrs.emplace_back();
      args_attrs.back().set(builder_.getIdentifier("xla_lhlo.params"),
                            builder_.getIndexAttr(alloc->parameter_number()));
    } else {
      block->addArgument(MemRefType::get({alloc->size()}, i8_type_));
      allocations_[alloc] = block->getArguments().back();
      args_attrs.emplace_back();
      args_attrs.back().set(builder_.getIdentifier("xla_lhlo.alloc"),
                            builder_.getIndexAttr(alloc->index()));
      if (alloc->maybe_live_out())
        args_attrs.back().set(builder_.getIdentifier("xla_lhlo.liveout"),
                              builder_.getBoolAttr(true));
    }
  }

  FunctionType function_type = builder_.getFunctionType(
      llvm::to_vector<8>(block->getArgumentTypes()), {});
  func_op.setType(function_type);
  func_op.setAllArgAttrs(args_attrs);

  SymbolTable symbol_table(module_);
  symbol_table.insert(func_op);
  builder_.setInsertionPointToEnd(block);

  const ::xla::HloInstructionSequence* schedule =
      assignment_.hlo_ordering().SequentialOrder(*computation);
  if (!schedule)
    return ::xla::Unimplemented("Missing sequential order for the computation");

  const std::vector<HloInstruction*>& ordering = schedule->instructions();
  TF_RETURN_IF_ERROR(computation->AcceptOrdered(this, ordering));
  builder_.create<ReturnOp>(builder_.getUnknownLoc());
  return Status::OK();
}

// Convert the MLIR `module` from HLO dialect to LHLO dialect using XLA for the
// given platform.
Status ConvertModule(ModuleOp module, StringRef platform_name) {
  SymbolTable symbol_table(module);
  if (!symbol_table.lookup("main")) {
    return ::xla::InvalidArgument(
        "conversion to HLO module failed: missing main()");
  }
  HloProto hlo_proto;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertMlirHloToHlo(module, &hlo_proto,
                          /*use_tuple_args=*/false,
                          /*return_tuple=*/false,
                          /*shape_representation_fn=*/nullptr),
      "conversion to XLA HLO proto failed");

  auto statusOrHloModule = HloModuleFromProto(hlo_proto);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(statusOrHloModule.status(),
                                  "parsing HLO proto to HLO module failed");
  std::unique_ptr<HloModule> hlo_module =
      std::move(statusOrHloModule.ValueOrDie());

  auto platform = ::xla::se::MultiPlatformManager::PlatformWithName(
      StringRefToView(platform_name));
  if (!platform.ok()) {
    std::string error_msg;
    llvm::raw_string_ostream os(error_msg);
    os << "failed to get platform: " << platform.status().ToString()
       << " (available Platform: ";
    std::vector<std::string> available_platforms;
    (void)::xla::se::MultiPlatformManager::PlatformsWithFilter(
        [&](const stream_executor::Platform* p) {
          available_platforms.push_back(p->Name());
          return false;
        });
    llvm::interleaveComma(available_platforms, os);
    os << ")";
    return ::xla::InvalidArgument("%s", os.str().c_str());
  }

  ::xla::BackendOptions backend_options;
  backend_options.set_platform(platform.ValueOrDie());
  auto backend_or_err = ::xla::Backend::CreateBackend(backend_options);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(backend_or_err.status(),
                                  "failed to create XLA Backend ");
  auto backend = std::move(backend_or_err.ValueOrDie());

  // Run all HLO passes to produce an optimized module.
  auto result_or = backend->compiler()->RunHloPassesAndBufferAssignement(
      std::move(hlo_module), backend->default_stream_executor(),
      backend->memory_allocator());
  TF_RETURN_WITH_CONTEXT_IF_ERROR(result_or.status(),
                                  "running XLA pass pipeline");
  std::unique_ptr<HloModule> optimized_hlo_module =
      std::move(std::get<0>(result_or.ValueOrDie()));
  std::unique_ptr<BufferAssignment> assignment =
      std::move(std::get<1>(result_or.ValueOrDie()));

  // Clear the module before populating it back with the result of the
  // conversion.
  module.getBody()->clear();
  OpBuilder builder(module);
  module.ensureTerminator(module.getBodyRegion(), builder, module.getLoc());

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      HloToLhloModule(*assignment, *optimized_hlo_module, module),
      "converting HLO to LHLO");

  return Status::OK();
}

// This pass take a MLIR HLO module, convert it to XLA to perform the HLO
// optimization pipeline for the required platform, and then convert back to
// MLIR LHLO.
class XlaHloToLhloPass
    : public PassWrapper<XlaHloToLhloPass, OperationPass<ModuleOp>> {
 public:
  XlaHloToLhloPass() = default;
  XlaHloToLhloPass(const XlaHloToLhloPass&) {}

 private:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    Status status = ConvertModule(module, platform_);
    if (!status.ok()) {
      module.emitError() << status.ToString();
      return signalPassFailure();
    }
  }

  Option<std::string> platform_{
      *this, "platform",
      llvm::cl::desc("The platform to use for the XLA optimization pipeline."),
      llvm::cl::init("Host")};
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createXlaHloToLhloWithXlaPass() {
  return std::make_unique<XlaHloToLhloPass>();
}

Status HloToLhloModule(const BufferAssignment& assignment,
                       const HloModule& hlo_module, ModuleOp module) {
  return LhloDialectEmitter(assignment, hlo_module, module).Run();
}

static PassRegistration<XlaHloToLhloPass> registration(
    "xla-hlo-to-lhlo-with-xla",
    "Emit LHLO from HLO using the existing XLA implementation");

}  // namespace mlir
