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
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
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
  // Populate the MLIR `module` with the computation from the `hlo_module` using
  // the provided buffer `assignment`. The returned `Status` indicates success
  // or failure in the conversion.
  static Status EmitModule(const BufferAssignment& assignment,
                           const HloModule& hlo_module, ModuleOp module) {
    return LhloDialectEmitter(assignment, hlo_module, module).Run();
  }

 private:
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

  Status DefaultAction(HloInstruction* hlo) final {
    return ::xla::Unimplemented("unsupported HLO %s", hlo->name());
  }

  // Computation parameters don't need any specific handling when they are
  // visited, they are already processed when we enter a new computation.
  Status HandleParameter(HloInstruction* instr) final { return Status::OK(); }

  // HLO Copy is translated 1:1 to an lhlo.copy operation.
  Status HandleCopy(HloInstruction* instr) final {
    TF_ASSIGN_OR_RETURN(Value source, GetOrCreateView(instr->operand(0)));
    TF_ASSIGN_OR_RETURN(Value dest, GetOrCreateView(instr));
    if (source != dest)
      builder_.create<xla_lhlo::CopyOp>(getLocation(instr),
                                        llvm::ArrayRef<Type>{}, source, dest);
    return Status::OK();
  }

  // Helper function to create view in a buffer for a given slice. The view is
  // cached in the `slices_` map.
  Value GetOrCreateView(const BufferAllocation::Slice& slice);

  // Helper function to create view in a buffer for a given instruction result.
  StatusOr<Value> GetOrCreateView(const HloInstruction* instr);

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

  // This map provides access to MLIR buffers for each HLO buffer slice. A slice
  // is contained in a BufferAllocation, and has an offset and a size.
  // The MLIR buffers are all `memref<{size}xi8>`. If the slice is the entire
  // BufferAllocation then the MLIR buffer corresponds to function
  // parameter for the allocation, otherwise it will map to a ViewOp in the
  // allocation. It is populated lazily in the `GetOrCreateView()` helper as we
  // process every instruction.
  using SliceKey = std::tuple<const BufferAllocation*, int64_t, int64_t>;
  llvm::DenseMap<SliceKey, Value> slices_;

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

Value LhloDialectEmitter::GetOrCreateView(
    const BufferAllocation::Slice& slice) {
  // Check if we already have a view for this slice, otherwise we need to create
  // a new one.
  SliceKey slice_key(slice.allocation(), slice.offset(), slice.size());
  auto slice_view_it = slices_.find(slice_key);
  if (slice_view_it != slices_.end()) return slice_view_it->second;

  // Check if we can just use the entire allocation before creating a view.
  Value alloc_buffer = allocations_[slice.allocation()];
  if (slice.offset() == 0 && slice.size() == slice.allocation()->size()) {
    slices_.insert({slice_key, alloc_buffer});
    return alloc_buffer;
  }

  // Create the view for this slice size, possible with an affine map to model
  // the offset. The result is cached in the slices_ map.
  SmallVector<AffineMap, 1> offset_map;
  if (slice.offset()) {
    offset_map.push_back(AffineMap::get(
        /*dimCount=*/1, /*symbolCount=*/0,
        {getAffineDimExpr(0, builder_.getContext()) + slice.offset()},
        builder_.getContext()));
  }
  auto slice_type = MemRefType::get({slice.size()}, i8_type_, offset_map);

  auto slice_view = builder_.create<ViewOp>(
      alloc_buffer.getLoc(), slice_type, alloc_buffer, /*operands=*/llvm::None);
  slices_.insert({slice_key, slice_view});
  return slice_view;
}

// Returns a view for the result of an instruction.
// We first get a view for the slice in the allocation, and then may need to
// create another view to adjust the slice for the shape of the instruction.
StatusOr<Value> LhloDialectEmitter::GetOrCreateView(
    const HloInstruction* instr) {
  const Shape& target_shape = instr->shape();
  TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice out_slice,
                      assignment_.GetUniqueTopLevelSlice(instr));
  Value slice_view = GetOrCreateView(out_slice);
  TF_ASSIGN_OR_RETURN(Type out_type, ::xla::ConvertShapeToType<MemRefType>(
                                         target_shape, builder_));
  if (slice_view.getType() != out_type)
    slice_view = builder_.create<ViewOp>(builder_.getUnknownLoc(), out_type,
                                         slice_view, llvm::None);
  return slice_view;
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

  // The function signature will be composed of:
  // - one memref for each of the parameters.
  // - one memref for each other buffer allocation.
  llvm::SmallVector<NamedAttributeList, 8> args_attrs;
  for (const HloInstruction* param : computation->parameter_instructions()) {
    TF_ASSIGN_OR_RETURN(auto arg_type, ::xla::ConvertShapeToType<MemRefType>(
                                           param->shape(), builder_));
    // First map parameters to memrefs on the operation.
    block->addArgument(arg_type);
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        assignment_.GetUniqueTopLevelSlice(param));
    allocations_[slice.allocation()] = block->getArguments().back();
    args_attrs.emplace_back();
    args_attrs.back().set(builder_.getIdentifier("xla_lhlo.params"),
                          builder_.getIndexAttr(param->parameter_number()));
  }

  for (const BufferAllocation& alloc : assignment_.Allocations()) {
    if (alloc.is_entry_computation_parameter()) continue;
    block->addArgument(MemRefType::get({alloc.size()}, i8_type_));
    allocations_[&alloc] = block->getArguments().back();
    args_attrs.emplace_back();
    args_attrs.back().set(builder_.getIdentifier("xla_lhlo.alloc"),
                          builder_.getIndexAttr(alloc.index()));
    if (alloc.maybe_live_out())
      args_attrs.back().set(builder_.getIdentifier("xla_lhlo.liveout"),
                            builder_.getBoolAttr(true));
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
      LhloDialectEmitter::EmitModule(*assignment, *optimized_hlo_module,
                                     module),
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

static PassRegistration<XlaHloToLhloPass> registration(
    "xla-hlo-to-lhlo-with-xla",
    "Emit LHLO from HLO using the existing XLA implementation");

}  // namespace mlir
