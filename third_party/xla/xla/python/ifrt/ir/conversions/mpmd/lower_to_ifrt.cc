/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/conversions/mpmd/lower_to_ifrt.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/conversions/mpmd/utils.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/built_in_spmd_expansions.h"
#include "xla/python/ifrt/ir/transforms/debug.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla::ifrt::mpmd {
namespace {

namespace mpmd = ::mlir::mpmd;

bool IsIfrtArray(mlir::Type t) { return mlir::isa<IfrtArrayType>(t); }

// Checks if a FuncOp is legal.
bool IsFuncOpLegal(mlir::func::FuncOp op) {
  if (mpmd::IsMainFunction(op)) {
    // All the inputs and results of the main func should be IFRT arrays.
    return absl::c_all_of(op.getFunctionType().getInputs(), IsIfrtArray) &&
           absl::c_all_of(op.getFunctionType().getResults(), IsIfrtArray);
  }
  // Any non-main function should not have mesh_shape attribute.
  return !op->hasAttr(mpmd::kMeshShapeAttr);
}

// Converts a Shardy MPMD MeshTensor to an IFRT Array.
//
// The conversion constructs an IFRT Array with the global shape of the
// Shardy MPMD MeshTensor, an IFRT Devices Attribute (i.e., list of IFRT devices
// corresponding to the MeshTensor's mesh), and an IFRT ShardingParam.
mlir::Type MeshTensorToArray(
    const llvm::StringMap<IfrtDevicesAttr>& mesh_name_to_devices_attr,
    mpmd::MeshTensorType mesh_tensor_type, mlir::sdy::MeshAttr mesh_attr) {
  mlir::MLIRContext* ctx = mesh_tensor_type.getContext();
  auto sharding_param =
      MeshTensorTypeToShardingParam(mesh_tensor_type, mesh_attr);
  CHECK_OK(sharding_param.status());
  return IfrtArrayType::get(
      ctx, mesh_tensor_type.getGlobalTensorType(),
      IfrtShardingParamAttr::get(ctx, sharding_param.value()),
      mesh_name_to_devices_attr.at(mesh_tensor_type.getMeshName()),
      mesh_tensor_type.getMemoryKind(), /*layout_attr=*/nullptr);
}

// Converts an MPMD TransferOp into an IFRT ReshardOp.
class TransferOpPattern final
    : public mlir::OpConversionPattern<mpmd::TransferOp> {
 public:
  TransferOpPattern(const mlir::TypeConverter& type_converter,
                    mlir::MLIRContext* context)
      : OpConversionPattern(type_converter, context) {}

  mlir::LogicalResult matchAndRewrite(
      mpmd::TransferOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const final {
    auto ifrt_reshard_op = ReshardOp::create(
        rewriter, op.getLoc(),
        /*outputs=*/getTypeConverter()->convertType(op.getType()),
        /*control_output=*/
        IfrtControlType::get(rewriter.getContext()),
        /*inputs=*/adaptor.getTensor(),
        /*donated=*/false,
        /*control_inputs=*/mlir::ValueRange());
    rewriter.replaceOp(op, ifrt_reshard_op.getOutputs());
    return mlir::success();
  }
};

// Converts an MPMD FragmentCallOp into an IFRT CallOp.
class FragmentCallOpPattern final
    : public mlir::OpConversionPattern<mpmd::FragmentCallOp> {
 public:
  FragmentCallOpPattern(
      const mlir::TypeConverter& type_converter, mlir::MLIRContext* context,
      const llvm::StringMap<IfrtDevicesAttr>& mesh_name_to_devices_attr)
      : OpConversionPattern(type_converter, context),
        mesh_name_to_devices_attr_(mesh_name_to_devices_attr) {}

  mlir::LogicalResult matchAndRewrite(
      mpmd::FragmentCallOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const final {
    // Convert the result types of the op.
    mlir::SmallVector<mlir::Type> converted_result_types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                converted_result_types))) {
      return mlir::failure();
    }

    // Get the aliased inputs from the callee in order to construct the
    // io_aliases required by the ifrt.CallOp.
    std::vector<mlir::Attribute> io_aliases;
    std::vector<int> donated_input_indices;
    mlir::func::FuncOp callee = mlir::cast<mlir::func::FuncOp>(
        mlir::cast<mlir::CallOpInterface>(*op).resolveCallable());
    for (mlir::BlockArgument arg : callee.getArguments()) {
      if (auto donation_attr = callee.getArgAttrOfType<mlir::BoolAttr>(
              arg.getArgNumber(), kBufferDonationAttrName)) {
        donated_input_indices.push_back(arg.getArgNumber());
      }
      if (auto aliasing_attr = callee.getArgAttrOfType<mlir::IntegerAttr>(
              arg.getArgNumber(), kAliasingOutputAttrName)) {
        io_aliases.push_back(rewriter.getDenseI32ArrayAttr(
            {static_cast<int32_t>(arg.getArgNumber()),
             static_cast<int32_t>(aliasing_attr.getInt())}));
      }
    }

    mlir::StringRef mesh_name = op.getMeshName();
    auto ifrt_call_op = CallOp::create(
        rewriter, op.getLoc(),
        /*outputs=*/converted_result_types,
        /*control_output=*/IfrtControlType::get(rewriter.getContext()),
        /*inputs=*/adaptor.getOperands(),
        /*control_inputs=*/mlir::ValueRange{},
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*callee=*/op.getCalleeAttr(),
        /*devices=*/mesh_name_to_devices_attr_.at(mesh_name),
        /*io_aliases=*/rewriter.getArrayAttr(io_aliases),
        /*donated_input_indices=*/
        rewriter.getDenseI32ArrayAttr(donated_input_indices));
    // Set the mesh name in an attribute. The mesh name is used to get optional
    // per-mesh compile options users might have provided.
    ifrt_call_op->setAttr(kIfrtMeshNameAttrName,
                          rewriter.getStringAttr(mesh_name));
    rewriter.replaceOp(op, ifrt_call_op.getOutputs());
    return mlir::success();
  }

 private:
  const llvm::StringMap<IfrtDevicesAttr>& mesh_name_to_devices_attr_;
};

// Pattern for converting the types of ReturnOps. It is used to ensure that the
// return of the main func is of type IFRT Array.
class ReturnOpPattern : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::ReturnOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

// Updates the main function signature from MeshTensor to IFRT Array, and
// removes mesh shapes attributes from non-main functions.
class FuncOpPattern final
    : public mlir::OpConversionPattern<mlir::func::FuncOp> {
 public:
  FuncOpPattern(const mlir::TypeConverter& type_converter,
                mlir::MLIRContext* context)
      : OpConversionPattern(type_converter, context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const final {
    if (!mpmd::IsMainFunction(op)) {
      // We only need to remove the mesh attributed for the non-main functions.
      rewriter.modifyOpInPlace(op, [&] { mpmd::RemoveMesh(op); });
      return mlir::success();
    }
    mlir::FunctionType func_type = op.getFunctionType();
    // Convert the function signature.
    mlir::TypeConverter::SignatureConversion converted_args(
        func_type.getNumInputs());
    const mlir::TypeConverter& type_converter = *getTypeConverter();
    if (failed(type_converter.convertSignatureArgs(func_type.getInputs(),
                                                   converted_args))) {
      return mlir::failure();
    }
    // Convert the function results.
    mlir::SmallVector<mlir::Type> converted_results;
    if (failed(type_converter.convertTypes(func_type.getResults(),
                                           converted_results))) {
      return mlir::failure();
    }
    // Replace the types of the region arguments as per the signature result.
    if (failed(rewriter.convertRegionTypes(&op.getBody(), type_converter,
                                           &converted_args))) {
      return mlir::failure();
    }
    // Update the function signature.
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(mlir::FunctionType::get(rewriter.getContext(),
                                         converted_args.getConvertedTypes(),
                                         converted_results));
      op->removeAttr(mpmd::kTopologyAttr);
      op->setAttr(kIfrtFunctionAttrName, rewriter.getUnitAttr());
    });
    return mlir::success();
  }
};

// Replaces aliasing and donation attributes with the IFRT donated arg
// attribute. The attributes are removed here because they will be removed
// implicitly during stable serialization (i.e., conversion to VIFRT), which
// requires the attributes to be from the IFRT dialect. Thus, we replace them
// with `ifrt.donated` unit attribute, which is supported by VIFRT.
void ReplaceAliasingArgAttrsWithIfrtDonatedArgAttrs(mlir::func::FuncOp func) {
  // Mark donated args with the IFRT donated unit attribute.
  for (int arg_num = 0; arg_num < func.getNumArguments(); arg_num++) {
    if (func.getArgAttrOfType<mlir::BoolAttr>(arg_num,
                                              kBufferDonationAttrName)) {
      func.removeArgAttr(arg_num, kBufferDonationAttrName);
      func.setArgAttr(arg_num, kIfrtDonatedArgAttrName,
                      mlir::UnitAttr::get(func.getContext()));
    } else if (func.getArgAttrOfType<mlir::IntegerAttr>(
                   arg_num, kAliasingOutputAttrName)) {
      func.removeArgAttr(arg_num, kAliasingOutputAttrName);
      func.setArgAttr(arg_num, kIfrtDonatedArgAttrName,
                      mlir::UnitAttr::get(func.getContext()));
    }
  }
}

class LowerToIfrtPass
    : public mlir::PassWrapper<LowerToIfrtPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToIfrtPass)

 private:
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<IfrtDialect>();
    AttachBuiltInSpmdExpansions(registry);
  }

  void runOnOperation() final {
    mlir::ModuleOp module_op = getOperation();
    mlir::MLIRContext& ctx = getContext();
    mlir::func::FuncOp main_func = GetMainFunction(module_op);

    ReplaceAliasingArgAttrsWithIfrtDonatedArgAttrs(main_func);

    // Construct mapping from mesh name to IFRT Device Attributes.
    llvm::StringMap<IfrtDevicesAttr> mesh_name_to_devices_attr;
    mlir::ArrayRef<mpmd::NamedMeshAttr> meshes =
        mpmd::GetTopologyMeshes(main_func);
    int total_devices = 0;
    for (const mpmd::NamedMeshAttr& mesh : meshes) {
      int num_mesh_devices = 1;
      for (mlir::sdy::MeshAxisAttr axis : mesh.getMesh().getAxes()) {
        num_mesh_devices *= axis.getSize();
      }
      mlir::SmallVector<int> mesh_devices(num_mesh_devices);
      absl::c_iota(mesh_devices, total_devices);
      total_devices += num_mesh_devices;
      mesh_name_to_devices_attr[mesh.getName()] =
          IfrtDevicesAttr::get(&ctx, mesh_devices);
    }

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<mpmd::MpmdDialect>();
    target.addLegalDialect<IfrtDialect, mlir::stablehlo::StablehloDialect,
                           mlir::func::FuncDialect, mlir::sdy::SdyDialect>();
    // The main func op should only have IFRT Array types, and the non-main
    // func ops should not have mesh_shape attribute.
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(IsFuncOpLegal);
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [](mlir::func::ReturnOp op) {
          return absl::c_all_of(op.getOperandTypes(), [](mlir::Type t) {
            return !mlir::isa<mpmd::MeshTensorType>(t);
          });
        });

    // Set conversion from MeshTensorType to IFRT Array.
    mlir::TypeConverter type_converter;
    type_converter.addConversion([&meshes, &mesh_name_to_devices_attr](
                                     mpmd::MeshTensorType mesh_tensor_type)
                                     -> mlir::Type {
      auto it = absl::c_find_if(
          meshes, [mesh_tensor_type](const mpmd::NamedMeshAttr& mesh) {
            return mesh.getName() == mesh_tensor_type.getMeshName();
          });
      CHECK(it != meshes.end())
          << "Mesh `" << mesh_tensor_type.getMeshName().str()
          << "` not found in topology.";
      return MeshTensorToArray(mesh_name_to_devices_attr, mesh_tensor_type,
                               it->getMesh());
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuncOpPattern, TransferOpPattern>(type_converter, &ctx);
    patterns.add<ReturnOpPattern>(&ctx);
    patterns.add<FragmentCallOpPattern>(type_converter, &ctx,
                                        mesh_name_to_devices_attr);
    if (mlir::failed(mlir::applyPartialConversion(module_op, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  mlir::StringRef getArgument() const override { return "mpmd-lower-to-ifrt"; }

  mlir::StringRef getDescription() const override {
    return "Lowers Shardy MPMD to IFRT IR.";
  }
};

class BuildCompileOptionsPass
    : public mlir::PassWrapper<BuildCompileOptionsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildCompileOptionsPass)

  explicit BuildCompileOptionsPass(
      CompileOptionsMap& compile_options_map,
      const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
          compile_options_overrides,
      int threshold_for_argument_tupling,
      llvm::function_ref<void(xla::ExecutableBuildOptions&, int64_t)>
          set_reserved_bytes)
      : compile_options_map_(compile_options_map),
        compile_options_overrides_(compile_options_overrides),
        threshold_for_argument_tupling_(threshold_for_argument_tupling),
        set_reserved_bytes_(set_reserved_bytes) {}

 private:
  CompileOptionsMap& compile_options_map_;
  const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
      compile_options_overrides_;
  int threshold_for_argument_tupling_;
  std::function<void(xla::ExecutableBuildOptions&, int64_t)>
      set_reserved_bytes_;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTableCollection symbol_table;
    mlir::func::FuncOp func_op = GetMainFunction(module);

    auto walk_result = func_op.walk([&](CallOp call_op) {
      mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
      mlir::ModuleOp callee_module = callee->getParentOfType<mlir::ModuleOp>();
      std::string callee_name = callee_module.getSymName()->str();

      xla::CompileOptions compile_options = GetDefaultCompileOptions(
          call_op, /*enable_sharding_propagation=*/false,
          /*enable_parameter_tupling=*/threshold_for_argument_tupling_ > 0 &&
              callee.getNumArguments() > threshold_for_argument_tupling_);

      if (auto reserved_hbm_bytes = callee->getAttrOfType<mlir::IntegerAttr>(
              mpmd::kReservedHbmBytes)) {
        set_reserved_bytes_(compile_options.executable_build_options,
                            reserved_hbm_bytes.getInt());
      };

      auto mesh_name_attr =
          call_op->getAttrOfType<mlir::StringAttr>(kIfrtMeshNameAttrName);
      if (mesh_name_attr == nullptr) {
        call_op.emitError()
            << " is missing " << kIfrtMeshNameAttrName.str() << " attribute";
        return mlir::WalkResult::interrupt();
      }
      // While the users provide per-mesh compilation options, we need to
      // include callee name in the key because fragments assigned to the same
      // mesh might have different `reserved_hbm_bytes`.
      const std::string compile_options_key =
          absl::StrCat(callee_name, "_mesh_", mesh_name_attr.str());
      call_op->setAttr(
          kIfrtCompileOptionsKey,
          mlir::StringAttr::get(call_op->getContext(), compile_options_key));
      // Apply the user-provided per-mesh compile option overrides.
      if (auto option_overrides =
              compile_options_overrides_.find(mesh_name_attr.str());
          option_overrides != compile_options_overrides_.end()) {
        compile_options.env_option_overrides = option_overrides->second;
      }
      compile_options_map_.emplace(compile_options_key, compile_options);
      return mlir::WalkResult::skip();
    });

    if (walk_result.wasInterrupted()) {
      signalPassFailure();
    }
  }

  mlir::StringRef getArgument() const override {
    return "mpmd-ifrt-build-compile-options";
  }

  mlir::StringRef getDescription() const override {
    return "Gets the compile options for each IFRT atom program.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerToIfrtPass() {
  return std::make_unique<LowerToIfrtPass>();
}

std::unique_ptr<mlir::Pass> CreateBuildCompileOptionsPass(
    CompileOptionsMap& compile_options_map,
    const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
        compile_options_overrides,
    int threshold_for_argument_tupling,
    llvm::function_ref<void(xla::ExecutableBuildOptions&, int64_t)>
        set_reserved_bytes) {
  return std::make_unique<BuildCompileOptionsPass>(
      compile_options_map, compile_options_overrides,
      threshold_for_argument_tupling, set_reserved_bytes);
}

void AddLowerToIfrtPasses(mlir::OpPassManager& pm) {
  pm.addPass(CreateLowerToIfrtPass());
  createIfrtToOutlinedAtomProgramsPipeline(pm);
}

void RegisterLowerToIfrtPasses() {
  mlir::registerPass(CreateLowerToIfrtPass);

  mlir::PassPipelineRegistration<> mpmd_lower_to_ifrt_pipeline(
      "ifrt-mpmd-lower-to-ifrt-pipeline", "Run the passes for lowering to ifrt",
      [](mlir::OpPassManager& pm) { AddLowerToIfrtPasses(pm); });
}

absl::Status LowerToIfrt(mlir::ModuleOp module) {
  mlir::func::FuncOp main_func = GetMainFunction(module);
  if (!mpmd::IsMpmdFunction(main_func)) {
    return absl::InvalidArgumentError("MLIR module is not an MPMD module.");
  }
  mlir::PassManager pm(module->getContext());
  InitPassManager(pm, "mpmd-lower-to-ifrt");
  pm.enableVerifier();

  AddLowerToIfrtPasses(pm);
  StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  if (mlir::failed(pm.run(module))) {
    return diagnostic_handler.ConsumeStatus();
  }
  return absl::OkStatus();
}

absl::StatusOr<CompileOptionsMap> GetCompileOptions(
    mlir::ModuleOp module,
    const absl::flat_hash_map<std::string, const EnvOptionsOverride>&
        compile_options_overrides,
    int threshold_for_argument_tupling,
    llvm::function_ref<void(xla::ExecutableBuildOptions&, int64_t)>
        set_reserved_bytes) {
  mlir::func::FuncOp main_func = GetMainFunction(module);
  if (!xla::ifrt::IsIfrtFunction(main_func)) {
    return absl::InvalidArgumentError("MLIR module is not an IFRT module.");
  }
  mlir::PassManager pm(module->getContext());
  InitPassManager(pm, "mpmd-ifrt-build-compile-options");
  CompileOptionsMap compile_options_map;
  pm.addPass(CreateBuildCompileOptionsPass(
      compile_options_map, compile_options_overrides,
      threshold_for_argument_tupling, set_reserved_bytes));
  StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  if (mlir::failed(pm.run(module))) {
    return diagnostic_handler.ConsumeStatus();
  }
  return compile_options_map;
}

}  // namespace xla::ifrt::mpmd
