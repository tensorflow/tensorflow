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

#include "tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h"

#include <climits>
#include <memory>
#include <tuple>

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/attribute_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/xla_mlir_translate_cl.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

using xla::BufferAllocation;
using xla::BufferAssignment;
using xla::HloComputation;
using xla::HloCustomCallInstruction;
using xla::HloInfeedInstruction;
using xla::HloInstruction;
using xla::HloModule;
using xla::HloModuleProto;
using xla::HloOutfeedInstruction;
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
  TF_ASSIGN_OR_RETURN(const xla::HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          module_proto, xla::GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(module_proto, module_config);
}

bool AllocationShouldLowerToTypedArg(const BufferAllocation* alloc) {
  return alloc->is_entry_computation_parameter() && !alloc->maybe_live_out();
}

}  // namespace

// Convert the MLIR `module` from HLO dialect to LHLO dialect using XLA for the
// given platform.
Status OptimizeAndConvertHloToLmhlo(std::unique_ptr<HloModule> hlo_module,
                                    ModuleOp module, StringRef platform_name) {
  auto platform = xla::se::MultiPlatformManager::PlatformWithName(
      StringRefToView(platform_name));
  if (!platform.ok()) {
    std::string error_msg;
    llvm::raw_string_ostream os(error_msg);
    os << "failed to get platform: " << platform.status().ToString()
       << " (available Platform: ";
    std::vector<std::string> available_platforms;
    (void)xla::se::MultiPlatformManager::PlatformsWithFilter(
        [&](const stream_executor::Platform* p) {
          available_platforms.push_back(p->Name());
          return false;
        });
    llvm::interleaveComma(available_platforms, os);
    os << ")";
    return xla::InvalidArgument("%s", os.str().c_str());
  }

  xla::BackendOptions backend_options;
  backend_options.set_platform(platform.ValueOrDie());
  auto backend_or_err = xla::Backend::CreateBackend(backend_options);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(backend_or_err.status(),
                                  "failed to create XLA Backend ");
  auto backend = std::move(backend_or_err.ValueOrDie());

  // Run all HLO passes to produce an optimized module.
  auto result_or = backend->compiler()->RunHloPassesAndBufferAssignement(
      std::move(hlo_module), backend->default_stream_executor(),
      optimize_xla_hlo, {backend->memory_allocator()});
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

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      HloToLhloModule(*assignment, *optimized_hlo_module, module),
      "converting HLO to LHLO");

  return Status::OK();
}

namespace {
// This pass takes an MLIR HLO module, converts it to XLA to perform the HLO
// optimization pipeline for the required platform, and then converts it back to
// MLIR LHLO.
class XlaHloToLhloPass
    : public PassWrapper<XlaHloToLhloPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<StandardOpsDialect, memref::MemRefDialect, mhlo::MhloDialect,
                lmhlo::LmhloDialect, lmhlo_gpu::LmhloGpuDialect>();
  }

 public:
  XlaHloToLhloPass() = default;
  XlaHloToLhloPass(const XlaHloToLhloPass&) {}
  StringRef getArgument() const final { return "xla-hlo-to-lhlo-with-xla"; }
  StringRef getDescription() const final {
    return "Emit LHLO from HLO using the existing XLA implementation";
  }

 private:
  void runOnOperation() final {
    ModuleOp module = getOperation();

    auto status = [&module, this]() -> Status {
      SymbolTable symbol_table(module);
      if (!symbol_table.lookup("main")) {
        return xla::InvalidArgument(
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

      return OptimizeAndConvertHloToLmhlo(std::move(hlo_module), module,
                                          platform_);
    }();
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

// Creates MLIR operands corresponding to operands and results of the XLA HLO
// instruction. If `num_operands` is valid, then only the first `num_operands`
// operands of the HLO instruction will be considered.
Status LhloDialectEmitter::CreateOperands(
    const HloInstruction* instr, absl::optional<xla::int64> num_operands,
    TokenLoweringMode token_mode, llvm::SmallVectorImpl<Value>& operands,
    size_t& num_arguments, size_t& num_results) {
  if (num_operands.value_or(0) > instr->operand_count())
    return xla::InvalidArgument("num_operands must be <= operand count");
  for (xla::int64 i = 0; i < num_operands.value_or(instr->operand_count());
       ++i) {
    TF_RETURN_IF_ERROR(GetOrCreateView(instr->operand(i), &operands,
                                       /*result_subset=*/{}, token_mode));
  }
  num_arguments = operands.size();
  TF_RETURN_IF_ERROR(
      GetOrCreateView(instr, &operands, /*result_subset=*/{}, token_mode));
  num_results = operands.size() - num_arguments;
  return Status::OK();
}

template <typename OpType>
OpType LhloDialectEmitter::CreateOpWithoutAttrs(const HloInstruction* instr,
                                                ValueRange operands) {
  Location loc = getLocation(instr);
  return builder_.create<OpType>(loc, llvm::None, operands,
                                 llvm::ArrayRef<NamedAttribute>{});
}

template <typename OpType>
StatusOr<OpType> LhloDialectEmitter::CreateOpWithoutAttrs(
    const HloInstruction* instr, size_t& num_arguments, size_t& num_results,
    absl::optional<xla::int64> num_operands) {
  llvm::SmallVector<Value, 4> operands;
  TF_RETURN_IF_ERROR(CreateOperands(instr, num_operands,
                                    TokenLoweringMode::kFailToLower, operands,
                                    num_arguments, num_results));
  return CreateOpWithoutAttrs<OpType>(instr, operands);
}

StatusOr<mlir::Operation*> LhloDialectEmitter::EmitOp(
    const HloInstruction* instr) {
  using xla::HloOpcode;
  switch (instr->opcode()) {
    case HloOpcode::kAbs:
      return CreateOpWithoutAttrs<lmhlo::AbsOp>(instr);
    case HloOpcode::kAdd:
      // HLO add ops on PRED elements are actually boolean or, but MHLO dialect
      // AddOps on i1 are just addition with overflow; so, we have to implement
      // the special behavior of HLO add ops on PRED here by creating an OrOp
      // instead.
      if (instr->shape().element_type() == xla::PRED) {
        return CreateOpWithoutAttrs<lmhlo::OrOp>(instr);
      } else {
        return CreateOpWithoutAttrs<lmhlo::AddOp>(instr);
      }
    case HloOpcode::kAddDependency:
      return nullptr;
    case HloOpcode::kAfterAll:
      // LMHLO is already ordered. This assumption may be broken after
      // introducing async regions and partial orders.
      return nullptr;
    case HloOpcode::kAllToAll:
      return EmitAllToAllOp(instr);
    case HloOpcode::kAllGather:
      return EmitAllGatherOp(instr);
    case HloOpcode::kAllReduce:
      return EmitAllReduceOp(instr);
    case HloOpcode::kAllReduceStart:
      return EmitAllReduceStartOp(instr);
    case HloOpcode::kAllReduceDone:
      return EmitAllReduceDoneOp(instr);
    case HloOpcode::kReduceScatter:
      return EmitReduceScatterOp(instr);
    case HloOpcode::kAnd:
      return CreateOpWithoutAttrs<lmhlo::AndOp>(instr);
    case HloOpcode::kAtan2:
      return CreateOpWithoutAttrs<lmhlo::Atan2Op>(instr);
    case HloOpcode::kBitcast:
      return EmitBitcast(instr);
    case HloOpcode::kBitcastConvert:
      return CreateOpWithoutAttrs<lmhlo::BitcastConvertOp>(instr);
    case HloOpcode::kBroadcast:
      return EmitBroadcastOp(instr);
    case HloOpcode::kCeil:
      return CreateOpWithoutAttrs<lmhlo::CeilOp>(instr);
    case HloOpcode::kCbrt:
      return CreateOpWithoutAttrs<lmhlo::CbrtOp>(instr);
    case HloOpcode::kClamp:
      return CreateOpWithoutAttrs<lmhlo::ClampOp>(instr);
    case HloOpcode::kCollectivePermute:
      return EmitCollectivePermuteOp(instr);
    case HloOpcode::kConditional:
      return EmitCaseOp(instr);
    case HloOpcode::kClz:
      return CreateOpWithoutAttrs<lmhlo::ClzOp>(instr);
    case HloOpcode::kCompare:
      return EmitCompareOp(instr);
    case HloOpcode::kComplex:
      return CreateOpWithoutAttrs<lmhlo::ComplexOp>(instr);
    case HloOpcode::kConcatenate:
      return EmitConcatenateOp(instr);
    case HloOpcode::kConvert:
      return CreateOpWithoutAttrs<lmhlo::ConvertOp>(instr);
    case HloOpcode::kCopy:
      return CreateOpWithoutAttrs<lmhlo::CopyOp>(instr);
    case HloOpcode::kCos:
      return CreateOpWithoutAttrs<lmhlo::CosOp>(instr);
    case HloOpcode::kDivide:
      return CreateOpWithoutAttrs<lmhlo::DivOp>(instr);
    case HloOpcode::kDot:
      return EmitDotOp(instr);
    case HloOpcode::kDynamicSlice:
      return EmitDynamicSliceOp(instr);
    case HloOpcode::kDynamicUpdateSlice:
      return CreateOpWithoutAttrs<lmhlo::DynamicUpdateSliceOp>(instr);
    case HloOpcode::kFft:
      return EmitFftOp(instr);
    case HloOpcode::kExp:
      return CreateOpWithoutAttrs<lmhlo::ExpOp>(instr);
    case HloOpcode::kExpm1:
      return CreateOpWithoutAttrs<lmhlo::Expm1Op>(instr);
    case HloOpcode::kFloor:
      return CreateOpWithoutAttrs<lmhlo::FloorOp>(instr);
    case HloOpcode::kGather:
      return EmitGatherOp(instr);
    case HloOpcode::kGetTupleElement:
      return nullptr;
    case HloOpcode::kImag:
      return CreateOpWithoutAttrs<lmhlo::ImagOp>(instr);
    case HloOpcode::kInfeed:
      return EmitInfeedOp(instr);
    case HloOpcode::kIota:
      return EmitIotaOp(instr);
    case HloOpcode::kIsFinite:
      return CreateOpWithoutAttrs<lmhlo::IsFiniteOp>(instr);
    case HloOpcode::kLog:
      return CreateOpWithoutAttrs<lmhlo::LogOp>(instr);
    case HloOpcode::kLog1p:
      return CreateOpWithoutAttrs<lmhlo::Log1pOp>(instr);
    case HloOpcode::kMap:
      return EmitMapOp(instr);
    case HloOpcode::kMaximum:
      return CreateOpWithoutAttrs<lmhlo::MaxOp>(instr);
    case HloOpcode::kMinimum:
      return CreateOpWithoutAttrs<lmhlo::MinOp>(instr);
    case HloOpcode::kMultiply:
      return CreateOpWithoutAttrs<lmhlo::MulOp>(instr);
    case HloOpcode::kNegate:
      return CreateOpWithoutAttrs<lmhlo::NegOp>(instr);
    case HloOpcode::kNot:
      return CreateOpWithoutAttrs<lmhlo::NotOp>(instr);
    case HloOpcode::kOr:
      return CreateOpWithoutAttrs<lmhlo::OrOp>(instr);
    case HloOpcode::kOutfeed:
      return EmitOutfeedOp(instr);
    case HloOpcode::kPartitionId:
      return CreateOpWithoutAttrs<lmhlo::PartitionIdOp>(instr);
    case HloOpcode::kPad:
      return EmitPadOp(instr);
    case HloOpcode::kPopulationCount:
      return CreateOpWithoutAttrs<lmhlo::PopulationCountOp>(instr);
    case HloOpcode::kPower:
      return CreateOpWithoutAttrs<lmhlo::PowOp>(instr);
    case HloOpcode::kReal:
      return CreateOpWithoutAttrs<lmhlo::RealOp>(instr);
    case HloOpcode::kReshape:
      return CreateOpWithoutAttrs<lmhlo::ReshapeOp>(instr);
    case HloOpcode::kReducePrecision:
      return EmitReducePrecisionOp(instr);
    case HloOpcode::kReduceWindow:
      return EmitReduceWindowOp(instr);
    case HloOpcode::kRemainder:
      return CreateOpWithoutAttrs<lmhlo::RemOp>(instr);
    case HloOpcode::kReplicaId:
      return CreateOpWithoutAttrs<lmhlo::ReplicaIdOp>(instr);
    case HloOpcode::kReverse:
      return EmitReverseOp(instr);
    case HloOpcode::kRoundNearestAfz:
      return CreateOpWithoutAttrs<lmhlo::RoundOp>(instr);
    case HloOpcode::kRsqrt:
      return CreateOpWithoutAttrs<lmhlo::RsqrtOp>(instr);
    case HloOpcode::kSelect:
      return CreateOpWithoutAttrs<lmhlo::SelectOp>(instr);
    case HloOpcode::kShiftLeft:
      return CreateOpWithoutAttrs<lmhlo::ShiftLeftOp>(instr);
    case HloOpcode::kShiftRightLogical:
      return CreateOpWithoutAttrs<lmhlo::ShiftRightLogicalOp>(instr);
    case HloOpcode::kShiftRightArithmetic:
      return CreateOpWithoutAttrs<lmhlo::ShiftRightArithmeticOp>(instr);
    case HloOpcode::kSign:
      return CreateOpWithoutAttrs<lmhlo::SignOp>(instr);
    case HloOpcode::kSin:
      return CreateOpWithoutAttrs<lmhlo::SinOp>(instr);
    case HloOpcode::kSlice:
      return EmitSliceOp(instr);
    case HloOpcode::kSqrt:
      return CreateOpWithoutAttrs<lmhlo::SqrtOp>(instr);
    case HloOpcode::kSubtract:
      return CreateOpWithoutAttrs<lmhlo::SubOp>(instr);
    case HloOpcode::kTanh:
      return CreateOpWithoutAttrs<lmhlo::TanhOp>(instr);
    case HloOpcode::kTranspose:
      return EmitTransposeOp(instr);
    case HloOpcode::kTriangularSolve:
      return EmitTriangularSolveOp(instr);
    case HloOpcode::kTuple:
      return nullptr;
    case HloOpcode::kXor:
      return CreateOpWithoutAttrs<lmhlo::XorOp>(instr);
    case HloOpcode::kSort:
      return EmitSortOp(instr);
    case HloOpcode::kFusion:
      return EmitFusionOp(instr);
    case HloOpcode::kScatter:
      return EmitScatterOp(instr);
    case HloOpcode::kSelectAndScatter:
      return EmitSelectAndScatterOp(instr);
    case HloOpcode::kCustomCall:
      return EmitCustomCallOp(instr);
    case HloOpcode::kConstant:
      return EmitConstant(instr);
    case HloOpcode::kReduce:
      return EmitReduceOp(instr);
    case HloOpcode::kRngGetAndUpdateState:
      return EmitRngGetAndUpdateStateOp(instr);
    case HloOpcode::kWhile:
      return EmitWhileOp(instr);
    default:
      llvm::errs() << instr->ToString();
      return tensorflow::errors::Internal(
          absl::StrCat("LHLO opcode ", xla::HloOpcodeString(instr->opcode()),
                       " is not supported."));
  }
}

Status LhloDialectEmitter::DefaultAction(const HloInstruction* instr) {
  return EmitOp(instr).status();
}

StatusOr<lmhlo::SortOp> LhloDialectEmitter::EmitSortOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto sort, CreateOpWithoutAttrs<lmhlo::SortOp>(instr));
  auto* sort_instr = xla::Cast<xla::HloSortInstruction>(instr);
  sort.dimensionAttr(builder_.getI64IntegerAttr(sort_instr->sort_dimension()));
  sort.is_stableAttr(builder_.getBoolAttr(sort_instr->is_stable()));
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *sort_instr->called_computations()[0], &sort.comparator(), &builder_));
  return sort;
}

// Walks MHLO::TupleOp recursively.
Status WalkTuplePostOrder(Value v,
                          const std::function<Status(Value)>& visitor) {
  if (auto* op = v.getDefiningOp()) {
    if (auto tuple = dyn_cast<mhlo::TupleOp>(op)) {
      for (Value sub_v : tuple.val()) {
        TF_RETURN_IF_ERROR(WalkTuplePostOrder(sub_v, visitor));
      }
      return Status::OK();
    }
  }
  return visitor(v);
}

StatusOr<Value> LhloDialectEmitter::RewriteFusionOperand(
    const HloInstruction* root, const Shape& shape,
    xla::ShapeIndex* shape_index, OpBuilder* b, Location loc) {
  if (shape.IsTuple()) {
    llvm::SmallVector<Value, 4> values;
    for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
      shape_index->push_back(i);
      TF_ASSIGN_OR_RETURN(
          auto v, RewriteFusionOperand(root, shape.tuple_shapes(i), shape_index,
                                       b, loc));
      values.push_back(v);
      shape_index->pop_back();
    }
    return Value(b->create<mhlo::TupleOp>(loc, values));
  }
  TF_ASSIGN_OR_RETURN(Value memref,
                      GetOrCreateArrayView(root, shape, *shape_index));
  auto load = b->create<memref::TensorLoadOp>(loc, memref);
  if (shape.layout() !=
      xla::LayoutUtil::MakeDescendingLayout(shape.dimensions().size())) {
    llvm::SmallVector<int64_t, 4> minor_to_major(
        shape.layout().minor_to_major().begin(),
        shape.layout().minor_to_major().end());
    load->setAttr("minor_to_major", GetLayoutAttribute(shape.layout(), b));
  }
  return load.getResult();
}

// Emit a lmhlo.fusion based on XLA HLO fusion. Structurally they are not neatly
// equivalent. Specifically, XLA HLO fusion:
//     fused_computation {
//       %p0 = parameter(0)
//       %p1 = parameter(1)
//       ...
//       ROOT %ret = ...
//     }
// will be converted to
//     lmhlo.fusion() {  // no explicit operands
//       // capturing outside buffers
//       %p0 = tensor_load(%arg0) : memref<...> -> tensor<...>
//       %p1 = tensor_load(%arg1) : memref<...> -> tensor<...>
//       ...
//       tensor_store ..., %ret // store a tensor to a memref
//     }
StatusOr<lmhlo::FusionOp> LhloDialectEmitter::EmitFusionOp(
    const HloInstruction* instr) {
  Location loc = getLocation(instr);

  auto* fusion_instr = xla::Cast<xla::HloFusionInstruction>(instr);

  auto fusion = builder_.create<lmhlo::FusionOp>(getLocation(instr));
  auto after_fusion = builder_.saveInsertionPoint();
  auto reverter = xla::MakeCleanup(
      [this, after_fusion] { builder_.restoreInsertionPoint(after_fusion); });
  builder_ = mlir::OpBuilder(fusion);

  auto region_builder = OpBuilder::atBlockBegin(&fusion.region().front());

  llvm::SmallVector<Value, 8> arguments;
  for (int i = 0; i < instr->operands().size(); ++i) {
    const HloInstruction* operand = instr->operand(i);
    xla::ShapeIndex shape_index;
    TF_ASSIGN_OR_RETURN(
        auto arg, RewriteFusionOperand(operand, operand->shape(), &shape_index,
                                       &region_builder, loc));
    arguments.push_back(arg);
  }

  TF_ASSIGN_OR_RETURN(Value result,
                      xla::HloFunctionImporter::ImportInstructions(
                          *fusion_instr->fused_instructions_computation(),
                          arguments, &region_builder));
  {
    int i = 0;
    llvm::SmallVector<Value, 4> output;
    TF_RETURN_IF_ERROR(GetOrCreateView(instr, &output));
    TF_RETURN_IF_ERROR(WalkTuplePostOrder(result, [&](Value v) mutable {
      region_builder.create<memref::TensorStoreOp>(loc, v, output[i++]);
      return Status::OK();
    }));
    if (i != output.size()) {
      return xla::InternalError("output sizes don't match");
    }
  }

  // Fold GTE/Tuple pairs.
  //
  // Since the fused region refers to values in its parent region, we can't
  // call applyPatternAndFoldGreedily. We optimize it manually.
  //
  // Only walk once, because post-ordering is exactly what we need for GTE
  // optimizations.
  fusion.region().walk([](mhlo::GetTupleElementOp gte) {
    SmallVector<Value, 4> folded_values;
    if (succeeded(OpBuilder(gte).tryFold(gte, folded_values))) {
      gte.replaceAllUsesWith(folded_values[0]);
    }
  });

  // Effectively a DCE on the region.
  {
    llvm::SmallVector<mlir::Operation*, 4> ops;
    fusion.region().walk([&](mlir::Operation* op) { ops.push_back(op); });
    // Visit the user first.
    std::reverse(ops.begin(), ops.end());
    for (auto op : ops) {
      if (isOpTriviallyDead(op)) op->erase();
    }
  }

  return fusion;
}

StatusOr<mhlo::ScatterDimensionNumbers>
LhloDialectEmitter::GetScatterDimensionNumbers(const HloInstruction* instr,
                                               mlir::MLIRContext* context) {
  auto* scatter_instr = xla::Cast<xla::HloScatterInstruction>(instr);

  const xla::ScatterDimensionNumbers& xla_scatter_dim =
      scatter_instr->scatter_dimension_numbers();

  mlir::Builder builder(context);
  auto get_i64_array_attr =
      [builder](absl::Span<const xla::int64> container) mutable {
        return builder.getI64TensorAttr(
            {container.data(), static_cast<size_t>(container.size())});
      };
  auto scatter_dimension_numbers = mhlo::ScatterDimensionNumbers::get(
      get_i64_array_attr(xla_scatter_dim.update_window_dims()),
      get_i64_array_attr(xla_scatter_dim.inserted_window_dims()),
      get_i64_array_attr(xla_scatter_dim.scatter_dims_to_operand_dims()),
      builder.getI64IntegerAttr(xla_scatter_dim.index_vector_dim()), context);
  return scatter_dimension_numbers;
}

StatusOr<lmhlo::ScatterOp> LhloDialectEmitter::EmitScatterOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto scatter,
                      CreateOpWithoutAttrs<lmhlo::ScatterOp>(instr));

  // copy attributes
  auto* scatter_instr = xla::Cast<xla::HloScatterInstruction>(instr);

  TF_ASSIGN_OR_RETURN(auto scatter_dimension_numbers,
                      GetScatterDimensionNumbers(instr, builder_.getContext()));
  scatter.scatter_dimension_numbersAttr(scatter_dimension_numbers);
  scatter.indices_are_sortedAttr(
      builder_.getBoolAttr(scatter_instr->indices_are_sorted()));
  scatter.unique_indicesAttr(
      builder_.getBoolAttr(scatter_instr->unique_indices()));

  // import update computation as region
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *scatter_instr->called_computations()[0], &scatter.update_computation(),
      &builder_));

  return scatter;
}

StatusOr<lmhlo::SelectAndScatterOp> LhloDialectEmitter::EmitSelectAndScatterOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto select_and_scatter,
                      CreateOpWithoutAttrs<lmhlo::SelectAndScatterOp>(instr));

  // copy attributes
  auto* select_and_scatter_instr =
      xla::Cast<xla::HloSelectAndScatterInstruction>(instr);
  const xla::Window& window = select_and_scatter_instr->window();

  if (xla::window_util::HasDilation(window)) {
    return xla::Unimplemented("Dilation for SelectAndScatter is not supported");
  }

  select_and_scatter.window_dimensionsAttr(
      GetWindowElements(window, [](const xla::WindowDimension& dim) {
        return static_cast<int64_t>(dim.size());
      }));
  select_and_scatter.window_stridesAttr(
      GetWindowElements(window, [](const xla::WindowDimension& dim) {
        return static_cast<int64_t>(dim.stride());
      }));
  select_and_scatter.paddingAttr(
      GetWindowElements(window, [](const xla::WindowDimension& dim) {
        return static_cast<int64_t>(dim.padding_low());
      }));

  // import select and scatter computation as region
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *select_and_scatter_instr->select(), &select_and_scatter.select(),
      &builder_));
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *select_and_scatter_instr->scatter(), &select_and_scatter.scatter(),
      &builder_));
  return select_and_scatter;
}

StatusOr<mlir::Operation*> LhloDialectEmitter::EmitCustomCallOp(
    const HloInstruction* instr) {
  auto* custom_call_instr = xla::Cast<xla::HloCustomCallInstruction>(instr);

  if (xla::gpu::IsCustomCallToCusolver(*instr)) {
    return EmitCholesky(custom_call_instr);
  }

  if (xla::gpu::IsCublasGemm(*instr)) {
    return EmitGemm(custom_call_instr);
  }

  if (xla::gpu::IsCustomCallToDnnConvolution(*instr)) {
    return EmitDnnConvolution(custom_call_instr);
  }

  if (xla::gpu::IsCustomCallToDnnBatchNorm(*instr)) {
    return EmitDnnBatchNorm(custom_call_instr);
  }

  // For custom call, if there are any token operands or results, they will not
  // be represented in LHLO so we need to remember the mapping. First create
  // operands where each token is replaced with a null Value.
  llvm::SmallVector<Value, 4> operands;
  size_t num_arguments, num_results;
  TF_RETURN_IF_ERROR(CreateOperands(instr, /*num_operands=*/absl::nullopt,
                                    TokenLoweringMode::kUseNull, operands,
                                    num_arguments, num_results));

  // Now check if any of the operands is Null, which would indicate the presence
  // of a token in the input or output.
  bool has_token = llvm::any_of(operands, [](Value v) { return !v; });

  lmhlo::CustomCallTargetArgMapping target_mapping;
  if (has_token) {
    // If there was a token, squeeze all the non-token arguments and results
    // (in-place) and remember the mapping.
    int next_index = 0;
    llvm::SmallVector<int64_t> arg_to_target_arg_mapping;
    for (int i = 0; i < num_arguments; ++i) {
      if (operands[i]) {
        arg_to_target_arg_mapping.push_back(i);
        operands[next_index++] = operands[i];
      }
    }
    // Size of arg_to_target_arg_mapping is the number of arguments in LHLO.
    llvm::SmallVector<int64_t> result_to_target_result_mapping;
    for (int i = num_arguments; i < operands.size(); ++i) {
      if (operands[i]) {
        result_to_target_result_mapping.push_back(i - num_arguments);
        operands[next_index++] = operands[i];
      }
    }

    // Build the mapping attribute.
    target_mapping = lmhlo::CustomCallTargetArgMapping::get(
        builder_.getI64IntegerAttr(num_arguments),
        builder_.getI64IntegerAttr(num_results),
        builder_.getI64ArrayAttr(arg_to_target_arg_mapping),
        builder_.getI64ArrayAttr(result_to_target_result_mapping),
        builder_.getContext());

    // Drop the remaining operands and adjust num_arguments and num_results
    // for LMHLO creation.
    operands.resize(next_index);
    num_arguments = arg_to_target_arg_mapping.size();
    num_results = result_to_target_result_mapping.size();
  }

  auto custom_call = CreateOpWithoutAttrs<lmhlo::CustomCallOp>(instr, operands);
  custom_call.call_target_nameAttr(
      builder_.getStringAttr(custom_call_instr->custom_call_target()));
  custom_call.backend_configAttr(
      builder_.getStringAttr(custom_call_instr->opaque()));
  const int32_t segments[2] = {static_cast<int32_t>(num_arguments),
                               static_cast<int32_t>(num_results)};
  custom_call->setAttr(lmhlo::CustomCallOp::getOperandSegmentSizeAttr(),
                       builder_.getI32VectorAttr(segments));
  if (target_mapping) custom_call.target_arg_mappingAttr(target_mapping);
  return custom_call.getOperation();
}

StatusOr<lmhlo_gpu::CholeskyOp> LhloDialectEmitter::EmitCholesky(
    const HloCustomCallInstruction* custom_call) {
  TF_ASSIGN_OR_RETURN(auto cholesky_op,
                      CreateOpWithoutAttrs<lmhlo_gpu::CholeskyOp>(custom_call));
  TF_ASSIGN_OR_RETURN(xla::CholeskyOptions options,
                      custom_call->backend_config<xla::CholeskyOptions>());
  cholesky_op.is_lowerAttr(builder_.getBoolAttr(options.lower()));
  return cholesky_op;
}

StatusOr<Operation*> LhloDialectEmitter::EmitGemm(
    const HloCustomCallInstruction* custom_call) {
  TF_ASSIGN_OR_RETURN(
      auto const config,
      custom_call->backend_config<xla::gpu::GemmBackendConfig>());

  auto set_common_attributes = [&](auto op) -> Operation* {
    auto hlo_dims = config.dot_dimension_numbers();
    auto mlir_dims = mhlo::DotDimensionNumbers::get(
        GetI64DenseElementsAttr(hlo_dims.lhs_batch_dimensions()),
        GetI64DenseElementsAttr(hlo_dims.rhs_batch_dimensions()),
        GetI64DenseElementsAttr(hlo_dims.lhs_contracting_dimensions()),
        GetI64DenseElementsAttr(hlo_dims.rhs_contracting_dimensions()),
        builder_.getContext());
    op.dot_dimension_numbersAttr(mlir_dims);
    op.alpha_realAttr(builder_.getF64FloatAttr(config.alpha_real()));
    op.alpha_imagAttr(builder_.getF64FloatAttr(config.alpha_imag()));
    op.batch_sizeAttr(builder_.getI64IntegerAttr(config.batch_size()));
    if (config.algorithm_case() ==
        xla::gpu::GemmBackendConfig::kSelectedAlgorithm) {
      op.algorithmAttr(builder_.getI64IntegerAttr(config.selected_algorithm()));
    }
    return op.getOperation();
  };

  if (custom_call->operand_count() == 2) {
    TF_ASSIGN_OR_RETURN(auto gemm,
                        CreateOpWithoutAttrs<lmhlo_gpu::GEMMOp>(custom_call));
    return set_common_attributes(gemm);
  }

  if (custom_call->operand_count() == 3) {
    TF_ASSIGN_OR_RETURN(
        auto gemm_bias,
        CreateOpWithoutAttrs<lmhlo_gpu::GEMM_BiasOp>(custom_call));
    gemm_bias.betaAttr(builder_.getF64FloatAttr(config.beta()));
    return set_common_attributes(gemm_bias);
  }

  return xla::InvalidArgument("GEMM custom call should have 2 or 3 operands");
}

static StatusOr<mlir::lmhlo_gpu::Activation> GetLHLOActivation(
    stream_executor::dnn::ActivationMode activation) {
  switch (activation) {
    case stream_executor::dnn::kNone:
      return mlir::lmhlo_gpu::Activation::None;
    case stream_executor::dnn::kSigmoid:
      return mlir::lmhlo_gpu::Activation::Sigmoid;
    case stream_executor::dnn::kRelu:
      return mlir::lmhlo_gpu::Activation::Relu;
    case stream_executor::dnn::kRelu6:
      return mlir::lmhlo_gpu::Activation::Relu6;
    case stream_executor::dnn::kReluX:
      return mlir::lmhlo_gpu::Activation::ReluX;
    case stream_executor::dnn::kTanh:
      return mlir::lmhlo_gpu::Activation::Tanh;
    case stream_executor::dnn::kBandPass:
      return mlir::lmhlo_gpu::Activation::BandPass;
    default:
      return xla::InternalError("Unknown activation");
  }
}

StatusOr<Operation*> LhloDialectEmitter::EmitDnnConvolution(
    const HloCustomCallInstruction* custom_call) {
  TF_ASSIGN_OR_RETURN(
      auto const backend_config,
      custom_call->backend_config<xla::gpu::CudnnConvBackendConfig>());

  TF_ASSIGN_OR_RETURN(const xla::gpu::CudnnConvKind kind,
                      xla::gpu::GetCudnnConvKind(custom_call));

  auto get_layout_attribute = [&](const xla::Layout& layout) {
    std::vector<int64_t> minor_to_major(layout.minor_to_major_size());
    absl::c_transform(layout.minor_to_major(), minor_to_major.begin(),
                      [](xla::int64 x) { return static_cast<int64_t>(x); });
    return builder_.getI64ArrayAttr(minor_to_major);
  };

  auto set_common_conv_attributes = [&, this](auto op) -> Operation* {
    const xla::Window& window = custom_call->window();
    // Window size for Cudnn Conv is same as the kernel size.
    op.window_stridesAttr(
        GetWindowElements(window, [](const xla::WindowDimension& dim) {
          return static_cast<int64_t>(dim.stride());
        }));
    // Cudnn Conv requires low and high padding to be equal.
    op.paddingAttr(
        GetWindowElements(window, [](const xla::WindowDimension& dim) {
          return static_cast<int64_t>(dim.padding_low());
        }));
    // LHS dilation is encoded in base_dilation of the backend config.
    // RHS dilation is encoded in window_dilation of the backend config.
    op.lhs_dilationAttr(
        GetWindowElements(window, [](const xla::WindowDimension& dim) {
          return static_cast<int64_t>(dim.base_dilation());
        }));
    op.rhs_dilationAttr(
        GetWindowElements(window, [](const xla::WindowDimension& dim) {
          return static_cast<int64_t>(dim.window_dilation());
        }));
    // Setup window reversal.
    auto window_reversal = llvm::to_vector<4>(llvm::map_range(
        window.dimensions(),
        [](const xla::WindowDimension& dim) { return dim.window_reversal(); }));
    auto type = RankedTensorType::get(op.window_strides()->getType().getShape(),
                                      builder_.getIntegerType(/*width=*/1));
    op.window_reversalAttr(DenseElementsAttr::get(type, window_reversal));

    op.dimension_numbersAttr(xla::ConvertConvDimensionNumbers(
        custom_call->convolution_dimension_numbers(), &builder_));
    op.feature_group_countAttr(
        builder_.getI64IntegerAttr(custom_call->feature_group_count()));
    op.batch_group_countAttr(
        builder_.getI64IntegerAttr(custom_call->batch_group_count()));
    op.precision_configAttr(xla::ConvertPrecisionConfig(
        &custom_call->precision_config(), &builder_));
    op.result_scaleAttr(
        builder_.getF64FloatAttr(backend_config.conv_result_scale()));
    auto config = mlir::lmhlo_gpu::ConvolutionBackendConfig::get(
        builder_.getI64IntegerAttr(backend_config.algorithm()),
        builder_.getBoolAttr(backend_config.tensor_ops_enabled()),
        get_layout_attribute(custom_call->operand(0)->shape().layout()),
        get_layout_attribute(custom_call->operand(1)->shape().layout()),
        get_layout_attribute(custom_call->shape().tuple_shapes(0).layout()),
        builder_.getContext());
    op.backend_configAttr(config);

    return op.getOperation();
  };

  auto set_activation = [&, this](auto op) -> Status {
    auto se_activation = static_cast<stream_executor::dnn::ActivationMode>(
        backend_config.activation_mode());
    TF_ASSIGN_OR_RETURN(mlir::lmhlo_gpu::Activation activation,
                        GetLHLOActivation(se_activation));
    StringAttr activation_attr = builder_.getStringAttr(
        mlir::lmhlo_gpu::stringifyActivation(activation));
    op.activation_modeAttr(activation_attr);
    return Status::OK();
  };

  switch (kind) {
    case xla::gpu::CudnnConvKind::kForward: {
      TF_ASSIGN_OR_RETURN(
          auto cnn_forward,
          CreateOpWithoutAttrs<lmhlo_gpu::ConvForwardOp>(custom_call));
      return set_common_conv_attributes(cnn_forward);
    }
    case xla::gpu::CudnnConvKind::kBackwardInput: {
      TF_ASSIGN_OR_RETURN(
          auto cnn_backward,
          CreateOpWithoutAttrs<lmhlo_gpu::ConvBackwardInputOp>(custom_call));
      return set_common_conv_attributes(cnn_backward);
    }
    case xla::gpu::CudnnConvKind::kBackwardFilter: {
      TF_ASSIGN_OR_RETURN(
          auto cnn_backward,
          CreateOpWithoutAttrs<lmhlo_gpu::ConvBackwardFilterOp>(custom_call));
      return set_common_conv_attributes(cnn_backward);
    }
    case xla::gpu::CudnnConvKind::kForwardActivation: {
      // Fused conv can be either with side input or without.
      if (custom_call->operand_count() == 3) {
        TF_ASSIGN_OR_RETURN(
            auto cnn_fused,
            CreateOpWithoutAttrs<lmhlo_gpu::ConvForwardFusedOp>(custom_call));
        TF_RETURN_IF_ERROR(set_activation(cnn_fused));
        return set_common_conv_attributes(cnn_fused);
      }

      TF_RET_CHECK(custom_call->operand_count() == 4);
      TF_ASSIGN_OR_RETURN(
          auto cnn_fused_side_input,
          CreateOpWithoutAttrs<lmhlo_gpu::ConvForwardFusedSideInputOp>(
              custom_call));
      cnn_fused_side_input.side_input_scaleAttr(
          builder_.getF64FloatAttr(backend_config.side_input_scale()));
      TF_RETURN_IF_ERROR(set_activation(cnn_fused_side_input));
      return set_common_conv_attributes(cnn_fused_side_input);
    }
  }
}

StatusOr<Operation*> LhloDialectEmitter::EmitDnnBatchNorm(
    const HloCustomCallInstruction* custom_call) {
  const xla::int64 num_operands = custom_call->operand_count();
  auto set_batchnorm_attributes = [&](auto op) -> StatusOr<Operation*> {
    // The last 2 operands of a custom call for batch norm are the epsilon and
    // feature_index.
    const HloInstruction* epsilon = custom_call->operand(num_operands - 2);
    TF_RET_CHECK(epsilon->IsConstant());
    float epsilon_value = epsilon->literal().Get<float>({});

    const HloInstruction* feature_index =
        custom_call->operand(num_operands - 1);
    TF_RET_CHECK(feature_index->IsConstant());
    xla::int64 feature_index_value =
        feature_index->literal().Get<xla::int64>({});

    op.epsilonAttr(builder_.getF32FloatAttr(epsilon_value));
    op.feature_indexAttr(builder_.getI64IntegerAttr(feature_index_value));
    return op.getOperation();
  };

  const std::string& target = custom_call->custom_call_target();
  if (target == xla::gpu::kCudnnBatchNormForwardTrainingCallTarget) {
    TF_ASSIGN_OR_RETURN(auto fwd_training,
                        CreateOpWithoutAttrs<lmhlo_gpu::BatchNormTrainingOp>(
                            custom_call, num_operands - 2));
    return set_batchnorm_attributes(fwd_training);
  }

  if (target == xla::gpu::kCudnnBatchNormBackwardCallTarget) {
    TF_ASSIGN_OR_RETURN(auto backward,
                        CreateOpWithoutAttrs<lmhlo_gpu::BatchNormGradOp>(
                            custom_call, num_operands - 2));
    return set_batchnorm_attributes(backward);
  }

  if (target == xla::gpu::kCudnnBatchNormForwardInferenceCallTarget) {
    TF_ASSIGN_OR_RETURN(auto fwd_inference,
                        CreateOpWithoutAttrs<lmhlo_gpu::BatchNormInferenceOp>(
                            custom_call, num_operands - 2));
    return set_batchnorm_attributes(fwd_inference);
  }

  return xla::Unimplemented("Unsupported batch norm operation");
}

// Convert an XLA HLO constant to a global_memref + get_global_memref pair.
StatusOr<mlir::memref::GetGlobalOp> LhloDialectEmitter::EmitConstant(
    const HloInstruction* instr) {
  // Insert a global_memref in the module.
  Location loc = getLocation(instr);

  auto const_instr = xla::Cast<xla::HloConstantInstruction>(instr);
  TF_RET_CHECK(const_instr->shape().IsArray() &&
               const_instr->shape().is_static());
  TF_ASSIGN_OR_RETURN(Type type, xla::ConvertShapeToType<MemRefType>(
                                     const_instr->shape(), builder_));
  auto memref_type = type.dyn_cast<MemRefType>();
  TF_RET_CHECK(memref_type != nullptr);

  TF_ASSIGN_OR_RETURN(
      DenseElementsAttr initial_value,
      CreateDenseElementsAttrFromLiteral(const_instr->literal(), builder_));

  std::string constant_name = xla::llvm_ir::ConstantNameToGlobalName(
      xla::llvm_ir::SanitizeConstantName(instr->name()));

  // Insert the global memref at the top level.
  {
    OpBuilder::InsertionGuard guard(builder_);
    builder_.clearInsertionPoint();
    auto global_var = builder_.create<memref::GlobalOp>(
        loc, constant_name, builder_.getStringAttr("private"), memref_type,
        initial_value, true);
    SymbolTable(module_).insert(global_var);
    global_var.getOperation()->moveBefore(&module_.front());

    // For operations that do not fold this constant value in their codegen, we
    // still need to materialize it into a buffer. Since buffer allocation is
    // already done, annotate the global_memref with the information to get to
    // the allocated buffer slice for this constant if need be.
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        assignment_.GetUniqueTopLevelSlice(instr));
    global_var->setAttr(
        "lmhlo.alloc",
        builder_.getIndexAttr(allocations_.find(slice.allocation())
                                  ->second.cast<BlockArgument>()
                                  .getArgNumber()));
    TF_RET_CHECK(slice.offset() == 0)
        << "Each constant should have its own allocation from BufferAssignment";
    TF_RET_CHECK(slice.allocation()->size() == slice.size())
        << "Each constant should have its own allocation from BufferAssignment";
  }

  auto get_global_memref =
      builder_.create<memref::GetGlobalOp>(loc, memref_type, constant_name);

  // Update the cache to remember this value.
  auto& cached_value = slices_[std::make_pair(instr, xla::ShapeIndex())];
  TF_RET_CHECK(cached_value == nullptr);
  cached_value = get_global_memref;
  return get_global_memref;
}

StatusOr<lmhlo::ReduceOp> LhloDialectEmitter::EmitReduceOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto reduce_op,
                      CreateOpWithoutAttrs<lmhlo::ReduceOp>(instr));
  auto* reduce = xla::Cast<xla::HloReduceInstruction>(instr);
  std::vector<int64_t> dimensions(reduce->dimensions().begin(),
                                  reduce->dimensions().end());
  reduce_op.dimensionsAttr(GetI64DenseElementsAttr(dimensions));
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *instr->called_computations()[0], &reduce_op.body(), &builder_));
  return reduce_op;
}

StatusOr<lmhlo::MapOp> LhloDialectEmitter::EmitMapOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto map_op, CreateOpWithoutAttrs<lmhlo::MapOp>(instr));
  auto* map = xla::Cast<xla::HloMapInstruction>(instr);
  std::vector<int64_t> dimensions(map->dimensions().begin(),
                                  map->dimensions().end());
  map_op.dimensionsAttr(GetI64DenseElementsAttr(dimensions));
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *instr->called_computations()[0], &map_op.computation(), &builder_));
  return map_op;
}

StatusOr<lmhlo::CompareOp> LhloDialectEmitter::EmitCompareOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto compare_op,
                      CreateOpWithoutAttrs<lmhlo::CompareOp>(instr));

  auto* compare = xla::Cast<xla::HloCompareInstruction>(instr);
  auto direction = [&]() {
    switch (compare->direction()) {
      case xla::ComparisonDirection::kEq:
        return mhlo::ComparisonDirection::EQ;
      case xla::ComparisonDirection::kNe:
        return mhlo::ComparisonDirection::NE;
      case xla::ComparisonDirection::kGe:
        return mhlo::ComparisonDirection::GE;
      case xla::ComparisonDirection::kGt:
        return mhlo::ComparisonDirection::GT;
      case xla::ComparisonDirection::kLe:
        return mhlo::ComparisonDirection::LE;
      case xla::ComparisonDirection::kLt:
        return mhlo::ComparisonDirection::LT;
    }
  }();
  compare_op.comparison_directionAttr(
      builder_.getStringAttr(stringifyComparisonDirection(direction)));
  auto compare_type = [&]() {
    switch (compare->type()) {
      case xla::Comparison::Type::kFloat:
        return mhlo::ComparisonType::FLOAT;
      case xla::Comparison::Type::kFloatTotalOrder:
        return mhlo::ComparisonType::TOTALORDER;
      case xla::Comparison::Type::kSigned:
        return mhlo::ComparisonType::SIGNED;
      case xla::Comparison::Type::kUnsigned:
        return mhlo::ComparisonType::UNSIGNED;
    }
  }();
  compare_op.compare_typeAttr(
      builder_.getStringAttr(stringifyComparisonType(compare_type)));
  return compare_op;
}

StatusOr<lmhlo::ReducePrecisionOp> LhloDialectEmitter::EmitReducePrecisionOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto reduce_precision_op,
                      CreateOpWithoutAttrs<lmhlo::ReducePrecisionOp>(instr));
  auto* reduce_precision = xla::Cast<xla::HloReducePrecisionInstruction>(instr);
  reduce_precision_op.exponent_bitsAttr(
      builder_.getI32IntegerAttr(reduce_precision->exponent_bits()));
  reduce_precision_op.mantissa_bitsAttr(
      builder_.getI32IntegerAttr(reduce_precision->mantissa_bits()));
  return reduce_precision_op;
}

namespace {
template <typename OpT>
void SetupChannelIdAttribute(OpT op, const xla::HloChannelInstruction* instr,
                             mlir::Builder builder) {
  if (instr->channel_id().has_value()) {
    op.channel_idAttr(mlir::mhlo::ChannelHandle::get(
        builder.getI64IntegerAttr(*instr->channel_id()),
        builder.getI64IntegerAttr(0), builder.getContext()));
  }
}

template <typename OpT>
Status SetupCommonCollectiveOpAttributes(OpT op, const HloInstruction* instr,
                                         mlir::OpBuilder& builder) {
  auto* collective = xla::Cast<xla::HloCollectiveInstruction>(instr);
  auto replica_groups_attr = xla::HloFunctionImporter::ConvertReplicaGroups(
      collective->replica_groups(), &builder);
  op->setAttr(replica_groups_attr.first, replica_groups_attr.second);
  op.constrain_layoutAttr(builder.getBoolAttr(collective->constrain_layout()));
  SetupChannelIdAttribute(op, collective, builder);
  return Status::OK();
}
}  // namespace

StatusOr<lmhlo::AllToAllOp> LhloDialectEmitter::EmitAllToAllOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto all_to_all_op,
                      CreateOpWithoutAttrs<lmhlo::AllToAllOp>(instr));
  auto* all_to_all = xla::Cast<xla::HloAllToAllInstruction>(instr);
  TF_RETURN_IF_ERROR(
      SetupCommonCollectiveOpAttributes(all_to_all_op, instr, builder_));
  if (all_to_all->split_dimension().has_value()) {
    all_to_all_op.split_dimensionAttr(
        builder_.getI64IntegerAttr(*all_to_all->split_dimension()));
  }
  return all_to_all_op;
}

StatusOr<lmhlo::AllGatherOp> LhloDialectEmitter::EmitAllGatherOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto all_gather_op,
                      CreateOpWithoutAttrs<lmhlo::AllGatherOp>(instr));
  auto* all_gather = xla::Cast<xla::HloAllGatherInstruction>(instr);
  TF_RETURN_IF_ERROR(
      SetupCommonCollectiveOpAttributes(all_gather_op, instr, builder_));
  all_gather_op.use_global_device_idsAttr(
      builder_.getBoolAttr(all_gather->use_global_device_ids()));
  all_gather_op.all_gather_dimensionAttr(
      builder_.getI64IntegerAttr(all_gather->all_gather_dimension()));
  return all_gather_op;
}

StatusOr<lmhlo::AllReduceOp> LhloDialectEmitter::EmitAllReduceOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto all_reduce_op,
                      CreateOpWithoutAttrs<lmhlo::AllReduceOp>(instr));
  auto* all_reduce = xla::Cast<xla::HloAllReduceInstruction>(instr);
  TF_RETURN_IF_ERROR(
      SetupCommonCollectiveOpAttributes(all_reduce_op, instr, builder_));
  all_reduce_op.use_global_device_idsAttr(
      builder_.getBoolAttr(all_reduce->use_global_device_ids()));
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *instr->called_computations()[0], &all_reduce_op.computation(),
      &builder_));
  return all_reduce_op;
}

StatusOr<lmhlo_gpu::AllReduceStartOp> LhloDialectEmitter::EmitAllReduceStartOp(
    const HloInstruction* instr) {
  llvm::SmallVector<Value, 4> operands;
  for (const HloInstruction* operand : instr->operands()) {
    TF_RETURN_IF_ERROR(GetOrCreateView(operand, &operands));
  }
  // Only include result index {1}. {0} always aliases the inputs.
  TF_RETURN_IF_ERROR(GetOrCreateView(instr, &operands, /*result_subset=*/{1}));

  Location loc = getLocation(instr);
  mlir::Type token_type = mlir::mhlo::TokenType::get(builder_.getContext());
  std::array<mlir::Type, 1> result_types = {token_type};
  lmhlo_gpu::AllReduceStartOp all_reduce_start_op =
      builder_.create<lmhlo_gpu::AllReduceStartOp>(loc, result_types, operands);

  auto* all_reduce = xla::Cast<xla::HloAllReduceInstruction>(instr);
  TF_RETURN_IF_ERROR(
      SetupCommonCollectiveOpAttributes(all_reduce_start_op, instr, builder_));
  all_reduce_start_op.use_global_device_idsAttr(
      builder_.getBoolAttr(all_reduce->use_global_device_ids()));
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *instr->called_computations()[0], &all_reduce_start_op.computation(),
      &builder_));

  TF_RET_CHECK(all_reduce_start_ops_.emplace(instr, all_reduce_start_op).second)
      << "all-reduce-start already lowered";
  return all_reduce_start_op;
}

StatusOr<lmhlo_gpu::AllReduceDoneOp> LhloDialectEmitter::EmitAllReduceDoneOp(
    const HloInstruction* instr) {
  auto it = all_reduce_start_ops_.find(instr->operand(0));
  TF_RET_CHECK(it != all_reduce_start_ops_.end())
      << "didn't find all-reduce-start op";

  llvm::SmallVector<Value, 4> operands;
  operands.push_back(it->second.token());
  all_reduce_start_ops_.erase(it);

  for (const HloInstruction* operand : instr->operands()) {
    TF_RETURN_IF_ERROR(GetOrCreateView(operand, &operands));
  }
  // We don't need to add buffers for the outputs, as these always alias inputs.
  return builder_.create<lmhlo_gpu::AllReduceDoneOp>(
      getLocation(instr), /*resultTypes=*/llvm::None, operands);
}

StatusOr<lmhlo::ReduceScatterOp> LhloDialectEmitter::EmitReduceScatterOp(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto reduce_scatter_op,
                      CreateOpWithoutAttrs<lmhlo::ReduceScatterOp>(instr));
  auto* ars = xla::Cast<xla::HloReduceScatterInstruction>(instr);
  TF_RETURN_IF_ERROR(
      SetupCommonCollectiveOpAttributes(reduce_scatter_op, instr, builder_));
  reduce_scatter_op.use_global_device_idsAttr(
      builder_.getBoolAttr(ars->use_global_device_ids()));
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *instr->called_computations()[0], &reduce_scatter_op.computation(),
      &builder_));
  reduce_scatter_op.scatter_dimensionAttr(
      builder_.getI64IntegerAttr(ars->scatter_dimension()));
  return reduce_scatter_op;
}

StatusOr<lmhlo::CollectivePermuteOp>
LhloDialectEmitter::EmitCollectivePermuteOp(const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto permute_op,
                      CreateOpWithoutAttrs<lmhlo::CollectivePermuteOp>(instr));
  auto* permute = xla::Cast<xla::HloCollectivePermuteInstruction>(instr);
  SetupChannelIdAttribute(permute_op, permute, builder_);
  mlir::NamedAttribute source_target_pairs_attr =
      xla::HloFunctionImporter::ConvertSourceTargetPairs(
          permute->source_target_pairs(), &builder_);
  permute_op->setAttr(source_target_pairs_attr.first,
                      source_target_pairs_attr.second);
  return permute_op;
}

StatusOr<lmhlo::InfeedOp> LhloDialectEmitter::EmitInfeedOp(
    const HloInstruction* instr) {
  const HloInfeedInstruction* infeed = xla::Cast<HloInfeedInstruction>(instr);
  // HLO Infeed instruction has a single operand of token type and a tuple
  // with buffers and a token as its output. LMHLO Infeed operation does not
  // need the token operand or result, so drop it.
  SmallVector<Value, 2> operands;
  TF_RETURN_IF_ERROR(GetOrCreateView(instr, &operands, /*result_subset=*/{0}));
  auto infeed_op = CreateOpWithoutAttrs<lmhlo::InfeedOp>(instr, operands);
  infeed_op.configAttr(builder_.getStringAttr(infeed->infeed_config()));
  return infeed_op;
}

StatusOr<lmhlo::OutfeedOp> LhloDialectEmitter::EmitOutfeedOp(
    const HloInstruction* instr) {
  const HloOutfeedInstruction* outfeed =
      xla::Cast<HloOutfeedInstruction>(instr);
  // HLO outfeed instruction has 2 operands, the source and a token, and a
  // single token output. LMHLO Outfeed does not need the token operand and
  // result, do drop it.
  SmallVector<Value, 2> operands;
  TF_RETURN_IF_ERROR(GetOrCreateView(instr->operand(0), &operands));
  auto outfeed_op = CreateOpWithoutAttrs<lmhlo::OutfeedOp>(instr, operands);
  outfeed_op.configAttr(builder_.getStringAttr(outfeed->outfeed_config()));
  return outfeed_op;
}

xla::StatusOr<lmhlo::BroadcastInDimOp> LhloDialectEmitter::EmitBroadcastOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto broadcast,
                      CreateOpWithoutAttrs<lmhlo::BroadcastInDimOp>(instr));
  broadcast.broadcast_dimensionsAttr(
      builder_.getI64TensorAttr(instr->dimensions()));
  return broadcast;
}

xla::StatusOr<lmhlo::ConcatenateOp> LhloDialectEmitter::EmitConcatenateOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto concat,
                      CreateOpWithoutAttrs<lmhlo::ConcatenateOp>(instr));
  auto hlo_concat = xla::Cast<xla::HloConcatenateInstruction>(instr);
  concat.dimensionAttr(
      builder_.getI64IntegerAttr(hlo_concat->concatenate_dimension()));
  return concat;
}

xla::StatusOr<lmhlo::IotaOp> LhloDialectEmitter::EmitIotaOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto iota, CreateOpWithoutAttrs<lmhlo::IotaOp>(instr));
  auto hlo_iota = xla::Cast<xla::HloIotaInstruction>(instr);
  iota.iota_dimensionAttr(
      builder_.getI64IntegerAttr(hlo_iota->iota_dimension()));
  return iota;
}

xla::StatusOr<lmhlo::ReverseOp> LhloDialectEmitter::EmitReverseOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto reverse,
                      CreateOpWithoutAttrs<lmhlo::ReverseOp>(instr));
  reverse.dimensionsAttr(builder_.getI64TensorAttr(instr->dimensions()));
  return reverse;
}

xla::StatusOr<lmhlo::TransposeOp> LhloDialectEmitter::EmitTransposeOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto transpose,
                      CreateOpWithoutAttrs<lmhlo::TransposeOp>(instr));
  transpose.permutationAttr(builder_.getI64TensorAttr(instr->dimensions()));
  return transpose;
}

xla::StatusOr<lmhlo::PadOp> LhloDialectEmitter::EmitPadOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto pad, CreateOpWithoutAttrs<lmhlo::PadOp>(instr));
  auto hlo_pad = xla::Cast<xla::HloPadInstruction>(instr);
  std::vector<xla::int64> low, high, interior;
  for (const auto& dim : hlo_pad->padding_config().dimensions()) {
    low.push_back(dim.edge_padding_low());
    high.push_back(dim.edge_padding_high());
    interior.push_back(dim.interior_padding());
  }
  pad.edge_padding_lowAttr(builder_.getI64TensorAttr(low));
  pad.edge_padding_highAttr(builder_.getI64TensorAttr(high));
  pad.interior_paddingAttr(builder_.getI64TensorAttr(interior));
  return pad;
}

xla::StatusOr<lmhlo::ReduceWindowOp> LhloDialectEmitter::EmitReduceWindowOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto reduce_window,
                      CreateOpWithoutAttrs<lmhlo::ReduceWindowOp>(instr));
  auto hlo_reduce_window = xla::Cast<xla::HloReduceWindowInstruction>(instr);
  std::vector<xla::int64> dims, strides, base_dilations, window_dilations,
      paddings;
  for (const auto& dim : hlo_reduce_window->window().dimensions()) {
    dims.push_back(dim.size());
    strides.push_back(dim.stride());
    base_dilations.push_back(dim.base_dilation());
    window_dilations.push_back(dim.window_dilation());
    paddings.push_back(dim.padding_low());
    paddings.push_back(dim.padding_high());
  }
  reduce_window.window_dimensionsAttr(builder_.getI64TensorAttr(dims));
  if (xla::window_util::HasStride(hlo_reduce_window->window())) {
    reduce_window.window_stridesAttr(builder_.getI64TensorAttr(strides));
  }
  if (xla::window_util::HasBaseDilation(hlo_reduce_window->window())) {
    reduce_window.base_dilationsAttr(builder_.getI64TensorAttr(base_dilations));
  }
  if (xla::window_util::HasWindowDilation(hlo_reduce_window->window())) {
    reduce_window.window_dilationsAttr(
        builder_.getI64TensorAttr(window_dilations));
  }
  CHECK_EQ(0, paddings.size() % 2);
  if (xla::window_util::HasPadding(hlo_reduce_window->window())) {
    reduce_window.paddingAttr(DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(paddings.size() / 2), 2},
                              builder_.getIntegerType(64)),
        paddings));
  }
  TF_RETURN_IF_ERROR(xla::HloFunctionImporter::ImportAsRegion(
      *hlo_reduce_window->called_computations()[0], &reduce_window.body(),
      &builder_));
  return reduce_window;
}

xla::StatusOr<lmhlo::SliceOp> LhloDialectEmitter::EmitSliceOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto slice, CreateOpWithoutAttrs<lmhlo::SliceOp>(instr));
  auto hlo_slice = xla::Cast<xla::HloSliceInstruction>(instr);
  slice.start_indicesAttr(builder_.getI64TensorAttr(hlo_slice->slice_starts()));
  slice.limit_indicesAttr(builder_.getI64TensorAttr(hlo_slice->slice_limits()));
  slice.stridesAttr(builder_.getI64TensorAttr(hlo_slice->slice_strides()));
  return slice;
}

xla::StatusOr<lmhlo::GatherOp> LhloDialectEmitter::EmitGatherOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto gather,
                      CreateOpWithoutAttrs<lmhlo::GatherOp>(instr));
  auto hlo_gather = xla::Cast<xla::HloGatherInstruction>(instr);
  gather.dimension_numbersAttr(xla::ConvertGatherDimensionNumbers(
      hlo_gather->gather_dimension_numbers(), &builder_));
  gather.slice_sizesAttr(builder_.getI64TensorAttr(
      std::vector<int64_t>(hlo_gather->gather_slice_sizes().begin(),
                           hlo_gather->gather_slice_sizes().end())));
  return gather;
}

xla::StatusOr<lmhlo::DynamicSliceOp> LhloDialectEmitter::EmitDynamicSliceOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto dynamic_slice,
                      CreateOpWithoutAttrs<lmhlo::DynamicSliceOp>(instr));
  auto hlo_dynamic_slice = xla::Cast<xla::HloDynamicSliceInstruction>(instr);
  dynamic_slice.slice_sizesAttr(
      builder_.getI64TensorAttr(hlo_dynamic_slice->dynamic_slice_sizes()));
  return dynamic_slice;
}

xla::StatusOr<lmhlo::DotOp> LhloDialectEmitter::EmitDotOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(auto dot, CreateOpWithoutAttrs<lmhlo::DotOp>(instr));
  auto hlo_dot = xla::Cast<xla::HloDotInstruction>(instr);
  dot.dot_dimension_numbersAttr(xla::ConvertDotDimensionNumbers(
      hlo_dot->dot_dimension_numbers(), &builder_));
  dot.precision_configAttr(
      xla::ConvertPrecisionConfig(&hlo_dot->precision_config(), &builder_));
  return dot;
}

xla::StatusOr<lmhlo::RngGetAndUpdateStateOp>
LhloDialectEmitter::EmitRngGetAndUpdateStateOp(
    const xla::HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(
      auto rng, CreateOpWithoutAttrs<lmhlo::RngGetAndUpdateStateOp>(instr));
  auto hlo_rng = xla::Cast<xla::HloRngGetAndUpdateStateInstruction>(instr);
  rng.deltaAttr(builder_.getI64IntegerAttr(hlo_rng->delta()));
  return rng;
}

xla::StatusOr<lmhlo::FftOp> LhloDialectEmitter::EmitFftOp(
    const HloInstruction* instr) {
  auto hlo_fft = xla::Cast<xla::HloFftInstruction>(instr);
  TF_ASSIGN_OR_RETURN(auto fft, CreateOpWithoutAttrs<lmhlo::FftOp>(instr));
  TF_ASSIGN_OR_RETURN(mlir::mhlo::FftType fft_type,
                      xla::ConvertFftType(hlo_fft->fft_type()));
  StringAttr fft_type_attr =
      builder_.getStringAttr(mlir::mhlo::stringifyFftType(fft_type));
  fft.fft_typeAttr(fft_type_attr);
  fft.fft_lengthAttr(GetI64DenseElementsAttr(instr->fft_length()));
  return fft;
}

xla::StatusOr<lmhlo::TriangularSolveOp>
LhloDialectEmitter::EmitTriangularSolveOp(const xla::HloInstruction* instr) {
  auto hlo_triangular_solve =
      xla::Cast<xla::HloTriangularSolveInstruction>(instr);
  TF_ASSIGN_OR_RETURN(auto triangular_solve,
                      CreateOpWithoutAttrs<lmhlo::TriangularSolveOp>(instr));
  const xla::TriangularSolveOptions& options =
      hlo_triangular_solve->triangular_solve_options();
  triangular_solve.left_sideAttr(builder_.getBoolAttr(options.left_side()));
  triangular_solve.lowerAttr(builder_.getBoolAttr(options.lower()));
  triangular_solve.unit_diagonalAttr(
      builder_.getBoolAttr(options.unit_diagonal()));
  TF_ASSIGN_OR_RETURN(mlir::mhlo::Transpose transpose,
                      xla::ConvertTranspose(options.transpose_a()));
  triangular_solve.transpose_aAttr(
      builder_.getStringAttr(mlir::mhlo::stringifyTranspose(transpose)));
  triangular_solve.layout_aAttr(
      GetLayoutAttribute(instr->operand(0)->shape().layout(), &builder_));
  triangular_solve.layout_bAttr(
      GetLayoutAttribute(instr->operand(1)->shape().layout(), &builder_));
  triangular_solve.layout_outputAttr(
      GetLayoutAttribute(instr->shape().layout(), &builder_));
  return triangular_solve;
}

xla::StatusOr<Operation*> LhloDialectEmitter::EmitBitcast(
    const xla::HloInstruction* instr) {
  // XLA buffer assignment should assign the same slice to a bitcast input and
  // output.
  const xla::ShapeIndex top_index;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      assignment_.GetUniqueSlice(instr, top_index));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice input_slice,
                      assignment_.GetUniqueSlice(instr->operand(0), top_index));

  if (input_slice != result_slice) {
    return xla::InvalidArgument(
        "Bitcast input and result slice should be same");
  }
  return nullptr;
}

mlir::DenseIntElementsAttr LhloDialectEmitter::GetLayoutAttribute(
    const xla::Layout& layout, Builder* builder) {
  llvm::SmallVector<int64_t, 4> minor_to_major(layout.minor_to_major().begin(),
                                               layout.minor_to_major().end());
  return builder->getIndexTensorAttr(minor_to_major);
}

Status LhloDialectEmitter::ImportAsLmhloRegion(xla::HloComputation* computation,
                                               mlir::Region* region) {
  auto after = builder_.saveInsertionPoint();
  auto reverter = xla::MakeCleanup(
      [this, after] { builder_.restoreInsertionPoint(after); });

  builder_ = OpBuilder(region);
  const xla::HloInstructionSequence* schedule =
      assignment_.hlo_ordering().SequentialOrder(*computation);
  if (!schedule)
    return xla::Unimplemented("Missing sequential order for the computation");
  TF_RETURN_IF_ERROR(
      computation->AcceptOrdered(this, schedule->instructions()));
  builder_.create<lmhlo::TerminatorOp>(builder_.getUnknownLoc());
  return Status::OK();
}

StatusOr<lmhlo::CaseOp> LhloDialectEmitter::EmitCaseOp(
    const HloInstruction* instr) {
  Location loc = getLocation(instr);
  llvm::SmallVector<Value, 4> operands;
  size_t num_arguments, num_results;
  TF_RETURN_IF_ERROR(CreateOperands(instr, 1, TokenLoweringMode::kUseNull,
                                    operands, num_arguments, num_results));

  auto case_op =
      builder_.create<lmhlo::CaseOp>(loc, operands[0], instr->branch_count());

  for (int i = 0; i < instr->branch_count(); i++) {
    case_op.branches()[i].push_back(new mlir::Block());
    TF_RETURN_IF_ERROR(ImportAsLmhloRegion(instr->called_computations()[i],
                                           &case_op.branches()[i]));
  }

  return case_op;
}

xla::StatusOr<lmhlo::WhileOp> LhloDialectEmitter::EmitWhileOp(
    const xla::HloInstruction* instr) {
  Location loc = getLocation(instr);
  SmallVector<Value, 1> operands;
  TF_RETURN_IF_ERROR(GetOrCreateView(
      instr->called_computations()[1]->root_instruction(), &operands));
  TF_RET_CHECK(operands.size() == 1);

  TF_ASSIGN_OR_RETURN(auto config,
                      instr->backend_config<xla::WhileLoopBackendConfig>());
  mlir::IntegerAttr trip_count;
  if (config.has_known_trip_count()) {
    trip_count = builder_.getI64IntegerAttr(config.known_trip_count().n());
  }
  lmhlo::WhileOp while_op =
      builder_.create<lmhlo::WhileOp>(loc, operands[0], trip_count);

  while_op.cond().push_back(new mlir::Block());
  while_op.body().push_back(new mlir::Block());
  TF_RETURN_IF_ERROR(
      ImportAsLmhloRegion(instr->called_computations()[1], &while_op.cond()));

  TF_RETURN_IF_ERROR(
      ImportAsLmhloRegion(instr->called_computations()[0], &while_op.body()));

  return while_op;
}

StatusOr<Value> LhloDialectEmitter::GetOrCreateArrayView(
    const xla::HloInstruction* instr, const xla::Shape& current_shape,
    const xla::ShapeIndex& shape_index) {
  // Cache generated ViewOp and StaticMemRefCastOp by (instruction,
  // shape_index).
  auto& cached_value = slices_[std::make_pair(instr, shape_index)];
  if (cached_value) {
    return cached_value;
  }

  if (instr->IsConstant() && shape_index.empty()) {
    TF_ASSIGN_OR_RETURN(Value constant_memref, EmitConstant(instr));
    return cached_value = constant_memref;
  }

  // If the shape happens to have dynamic dimensions, create the memref using
  // the underlying static shape.
  // TODO(jurahul): Revisit this when we can model memrefs with dynamic shape
  // but static bounds in MLIR.
  const Shape static_shape = xla::ShapeUtil::MakeStaticShape(current_shape);

  TF_ASSIGN_OR_RETURN(Type out_type, xla::ConvertShapeToType<MemRefType>(
                                         static_shape, builder_));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                      assignment_.GetUniqueSlice(instr, shape_index));
  Value alloc = allocations_[slice.allocation()];

  // TODO(timshen): revisit location handling.
  Location loc = builder_.getUnknownLoc();

  Value result;
  if (AllocationShouldLowerToTypedArg(slice.allocation())) {
    TF_RET_CHECK(slice.offset() == 0);
    TF_RET_CHECK(slice.size() == slice.allocation()->size());
    result = alloc;
  } else {
    Value byte_shift =
        builder_.create<ConstantIndexOp>(alloc.getLoc(), slice.offset());

    xla::Shape physical_shape =
        xla::ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
            static_shape);
    TF_ASSIGN_OR_RETURN(
        Type physical_out_type,
        xla::ConvertShapeToType<MemRefType>(physical_shape, builder_));

    // ViewOp only takes memrefs without affine maps (layouts). Let ViewOp
    // produce the physical shape (where dimensions are ordered in major to
    // minor) first, then follow up with a MemRefReinterpretCast to cast the
    // resulting memref to the original layout.
    result = builder_.create<memref::ViewOp>(loc, physical_out_type, alloc,
                                             byte_shift,
                                             /*sizes=*/ValueRange{});
  }
  if (result.getType() != out_type) {
    int64_t out_offset;
    SmallVector<int64_t, 4> out_strides;
    auto out_memref_type = out_type.dyn_cast<MemRefType>();
    if (!out_memref_type)
      return tensorflow::errors::Internal(
          "Expected memref type when creating a view for leaf type of a "
          "tuple.");
    if (failed(getStridesAndOffset(out_memref_type, out_strides, out_offset)))
      return tensorflow::errors::Internal(
          "Failed to get strides and offset from the output type.");
    result = builder_.create<memref::ReinterpretCastOp>(
        loc, out_memref_type, result, out_offset, out_memref_type.getShape(),
        out_strides);
  }
  return cached_value = result;
}

Status LhloDialectEmitter::GetOrCreateViewImpl(
    const HloInstruction* instr, const Shape& current_shape,
    xla::ShapeIndex* current_shape_index, SmallVectorImpl<Value>* values,
    TokenLoweringMode token_mode) {
  if (current_shape.IsTuple()) {
    for (int i = 0; i < current_shape.tuple_shapes().size(); ++i) {
      current_shape_index->push_back(i);
      TF_RETURN_IF_ERROR(
          GetOrCreateViewImpl(instr, current_shape.tuple_shapes(i),
                              current_shape_index, values, token_mode));
      current_shape_index->pop_back();
    }
    return Status::OK();
  }
  if (current_shape.IsArray()) {
    TF_ASSIGN_OR_RETURN(auto v, GetOrCreateArrayView(instr, current_shape,
                                                     *current_shape_index));
    values->push_back(v);
    return Status::OK();
  }
  if (current_shape.IsToken()) {
    switch (token_mode) {
      case TokenLoweringMode::kFailToLower:
        return xla::InternalError(
            "Unexpected token kind for %s and shape index %s",
            instr->ToString(), current_shape_index->ToString());

      case TokenLoweringMode::kUseNull:
        values->push_back(Value{});
        return Status::OK();
    }
  }
  return xla::InternalError("Unexpected shape kind for %s and shape index %s",
                            instr->ToString(), current_shape_index->ToString());
}

// Returns a view for the result of an instruction.
// We first get a view for the slice in the allocation, and then may need to
// create another view to adjust the slice for the shape of the instruction.
Status LhloDialectEmitter::GetOrCreateView(const HloInstruction* instr,
                                           SmallVectorImpl<Value>* values,
                                           const xla::ShapeIndex& result_subset,
                                           TokenLoweringMode token_mode) {
  xla::ShapeIndex shape_index = result_subset;
  const Shape& sub_shape =
      xla::ShapeUtil::GetSubshape(instr->shape(), shape_index);
  return GetOrCreateViewImpl(instr, sub_shape, &shape_index, values,
                             token_mode);
}

Status LhloDialectEmitter::Initialize() {
  TF_RET_CHECK(computation_.IsEntryComputation());

  mlir::IntegerAttr unique_id =
      builder_.getI32IntegerAttr(computation_.parent()->unique_id());
  module_->setAttr("hlo.unique_id", unique_id);
  std::string function_name =
      computation_.name().empty() ? "__compute" : computation_.name();

  // Create the function as () -> (), we'll compute the arguments from the
  // buffer allocation and update the type then.
  auto func_op = FuncOp::create(builder_.getUnknownLoc(), function_name,
                                builder_.getFunctionType({}, {}));

  {
    // This is an optional attribute used by the XLA backend. If the resulting
    // LMHLO doesn't go through XLA, this is not needed.
    const Shape& shape = computation_.root_instruction()->shape();
    func_op->setAttr(
        "result_xla_shape",
        builder_.getStringAttr(shape.ToString(/*print_layout=*/true)));
  }
  Block* block = func_op.addEntryBlock();

  llvm::SmallVector<const BufferAllocation*, 8> ordered_allocations;
  for (const BufferAllocation& alloc : assignment_.Allocations())
    ordered_allocations.push_back(&alloc);

  if (computation_.IsEntryComputation()) {
    // Sort the rather arbitrarily ordered allocations to match the input/output
    // parameters. Specifically we want to sort buffer allocations in the
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
        return std::tuple<int, const xla::ShapeIndex&>(
                   lhs->parameter_number(), lhs->param_shape_index()) <
               std::tuple<int, const xla::ShapeIndex&>(
                   rhs->parameter_number(), rhs->param_shape_index());
      }
      return false;
    };

    std::stable_sort(ordered_allocations.begin(), ordered_allocations.end(),
                     allocation_comparator);
  }

  absl::flat_hash_map<const BufferAllocation*,
                      std::pair<const Shape*, xla::ShapeIndex>>
      allocation_to_output_info;
  TF_RETURN_IF_ERROR(xla::ShapeUtil::ForEachSubshapeWithStatus(
      computation_.root_instruction()->shape(),
      [&](const Shape& sub_shape, xla::ShapeIndex index) -> Status {
        TF_ASSIGN_OR_RETURN(
            auto slice,
            assignment_.GetUniqueSlice(computation_.root_instruction(), index));
        const BufferAllocation* alloc = slice.allocation();
        TF_RET_CHECK(slice.offset() == 0);
        TF_RET_CHECK(slice.size() == alloc->size());
        allocation_to_output_info[alloc] = std::make_pair(&sub_shape, index);
        return Status::OK();
      }));

  // The function signature will be composed of:
  // - one memref for each of the parameters.
  // - one memref for each other buffer allocation.
  llvm::SmallVector<DictionaryAttr, 8> args_attrs;
  for (const BufferAllocation* alloc : ordered_allocations) {
    if (alloc->is_thread_local()) {
      continue;
    }

    // There are optional attributes to help the program run through XLA. XLA
    // defines ExecutionInput and ExecutionOutput structures to carry
    // input-output type and buffer information, therefore any information they
    // need (mainly the type structure, potentially containing tuples) to be
    // preserved. They are not needed if the generated LMHLO is not sent to XLA.
    NamedAttrList arg_attr_list;
    mlir::Type arg_type;
    if (AllocationShouldLowerToTypedArg(alloc)) {
      xla::Shape buffer_shape = xla::ShapeUtil::GetSubshape(
          computation_.parameter_instruction(alloc->parameter_number())
              ->shape(),
          alloc->param_shape_index());

      if (buffer_shape.IsTuple()) {
        arg_type = MemRefType::get({alloc->size()}, i8_type_);
      } else {
        // TODO(jurahul): Revisit this when we can model memrefs with dynamic
        // shape but static bounds in MLIR.
        const Shape static_shape =
            xla::ShapeUtil::MakeStaticShape(buffer_shape);
        TF_ASSIGN_OR_RETURN(arg_type, xla::ConvertShapeToType<MemRefType>(
                                          static_shape, builder_));
      }
    } else {
      arg_type = MemRefType::get({alloc->size()}, i8_type_);
    }

    if (alloc->is_entry_computation_parameter()) {
      arg_attr_list.set("lmhlo.params",
                        builder_.getIndexAttr(alloc->parameter_number()));
      if (!alloc->param_shape_index().empty()) {
        arg_attr_list.set("lmhlo.param_shape_index",
                          builder_.getI64TensorAttr(llvm::makeArrayRef(
                              alloc->param_shape_index().begin(),
                              alloc->param_shape_index().end())));
      }
    }
    // Optional: an attribute for optimization. If a kernel uses this
    // allocation, but the allocation has lmhlo.constant_name, then the kernel
    // will instead use the global value indicated by the name for potentially
    // more optimizations (e.g. constant propagation).
    if (alloc->is_constant()) {
      arg_attr_list.set(
          "lmhlo.constant_name",
          builder_.getStringAttr(
              xla::llvm_ir::ConstantBufferAllocationToGlobalName(*alloc)));
    }
    auto iter = allocation_to_output_info.find(alloc);
    if (iter != allocation_to_output_info.end()) {
      const Shape* sub_shape = iter->second.first;
      const xla::ShapeIndex& shape_index = iter->second.second;
      if (!sub_shape->IsArray()) {
        continue;
      }
      arg_attr_list.set("lmhlo.output_index",
                        builder_.getI64TensorAttr(llvm::makeArrayRef(
                            shape_index.begin(), shape_index.end())));
      if (auto alias = computation_.parent()
                           ->input_output_alias_config()
                           .GetAliasedParameter(shape_index)) {
        if (alias->must_alias()) {
          arg_attr_list.set("lmhlo.must_alias", builder_.getUnitAttr());
        }
      }
    }
    block->addArgument(arg_type);
    allocations_[alloc] = block->getArguments().back();
    args_attrs.push_back(arg_attr_list.getDictionary(builder_.getContext()));
  }

  FunctionType function_type =
      builder_.getFunctionType(block->getArgumentTypes(), {});
  func_op.setType(function_type);
  func_op.setAllArgAttrs(args_attrs);

  SymbolTable symbol_table(module_);
  symbol_table.insert(func_op);
  builder_.setInsertionPointToEnd(block);

  auto return_op =
      builder_.create<lmhlo::TerminatorOp>(builder_.getUnknownLoc());
  builder_ = OpBuilder(return_op);

  return Status::OK();
}

std::unique_ptr<OperationPass<ModuleOp>> createXlaHloToLhloWithXlaPass() {
  return std::make_unique<XlaHloToLhloPass>();
}

Status HloToLhloModule(const BufferAssignment& assignment,
                       const HloModule& hlo_module, ModuleOp module) {
  module.getContext()
      ->loadDialect<StandardOpsDialect, memref::MemRefDialect,
                    mhlo::MhloDialect, lmhlo::LmhloDialect,
                    lmhlo_gpu::LmhloGpuDialect>();

  module->setLoc(mlir::NameLoc::get(
      mlir::Identifier::get(hlo_module.name(), module.getContext())));

  const HloComputation* computation = hlo_module.entry_computation();

  LhloDialectEmitter emitter(assignment, *computation, module);
  TF_RETURN_IF_ERROR(emitter.Initialize());

  const xla::HloInstructionSequence* schedule =
      assignment.hlo_ordering().SequentialOrder(*computation);
  if (!schedule)
    return xla::Unimplemented("Missing sequential order for the computation");
  const std::vector<HloInstruction*>& ordering = schedule->instructions();
  TF_RETURN_IF_ERROR(computation->AcceptOrdered(&emitter, ordering));
  TF_RET_CHECK(succeeded(mlir::verify(module)));
  return Status::OK();
}

OwningModuleRef HloTextToLhloTranslateFunction(llvm::StringRef input,
                                               MLIRContext* context) {
  StatusOr<std::unique_ptr<HloModule>> maybe_module =
      xla::ParseAndReturnUnverifiedModule(
          absl::string_view(input.data(), input.size()));
  TF_CHECK_OK(maybe_module.status());

  OwningModuleRef module = ModuleOp::create(UnknownLoc::get(context));

  TF_CHECK_OK(OptimizeAndConvertHloToLmhlo(maybe_module.ConsumeValueOrDie(),
                                           module.get(), "Host"));

  return module;
}

static PassRegistration<XlaHloToLhloPass> registration;

}  // namespace mlir
