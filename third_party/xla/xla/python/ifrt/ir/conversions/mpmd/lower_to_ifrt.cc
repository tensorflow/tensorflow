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
#include "llvm/ADT/STLExtras.h"
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
#include "xla/service/compilation_environments.h"
#include "xla/service/computation_placer.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla::ifrt::mpmd {
namespace {

using llvm::DenseMap;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::BlockArgument;
using mlir::CallOpInterface;
using mlir::ConversionPatternRewriter;
using mlir::DenseSet;
using mlir::failed;
using mlir::failure;
using mlir::FunctionType;
using mlir::IntegerAttr;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpConversionPattern;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::SmallVector;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::success;
using mlir::SymbolTableCollection;
using mlir::Type;
using mlir::TypeConverter;
using mlir::TypedValue;
using mlir::ValueRange;
using mlir::WalkResult;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::mpmd::FragmentCallOp;
using mlir::mpmd::GetMainFunction;
using mlir::mpmd::IsMainFunction;
using mlir::mpmd::kAliasingAttrName;
using mlir::mpmd::kBufferDonationAttrName;
using mlir::mpmd::kReservedHbmBytes;
using mlir::mpmd::MeshTensorType;
using mlir::mpmd::MpmdDialect;
using mlir::mpmd::RemoveMesh;
using mlir::mpmd::TransferOp;
using xla::ifrt::IfrtControlType;
using xla::ifrt::IfrtDevicesAttr;
using xla::ifrt::InitPassManager;
using xla::ifrt::StatusScopedDiagnosticHandler;

namespace mpmd = ::mlir::mpmd;
namespace sdy = mlir::sdy;

bool IsIfrtArray(Type t) { return mlir::isa<xla::ifrt::IfrtArrayType>(t); }

// Checks if a FuncOp is legal.
bool IsFuncOpLegal(FuncOp op) {
  if (IsMainFunction(op)) {
    // All the inputs and results of the main func should be IFRT arrays.
    return absl::c_all_of(op.getFunctionType().getInputs(), IsIfrtArray) &&
           absl::c_all_of(op.getFunctionType().getResults(), IsIfrtArray);
  }
  // Any non-main function should not have mesh_shape attribute.
  return !op->hasAttr(mpmd::kMeshShapeAttr);
}

// Converts a PartIR:MPMD MeshTensor to an IFRT Array.
//
// The conversion constructs an IFRT Array with the global shape of the
// PartIR:MPMD MeshTensor, an IFRT Devices Attribute (i.e., list of IFRT devices
// corresponding to the MeshTensor's mesh), and an IFRT ShardingParam.
Type MeshTensorToArray(
    const llvm::StringMap<IfrtDevicesAttr>& mesh_name_to_devices_attr,
    MeshTensorType mesh_tensor_type, sdy::MeshAttr mesh_attr) {
  MLIRContext& ctx = *mesh_tensor_type.getContext();
  auto sharding_param =
      MeshTensorTypeToShardingParam(mesh_tensor_type, mesh_attr);
  CHECK_OK(sharding_param.status());
  return xla::ifrt::IfrtArrayType::get(
      &ctx, mesh_tensor_type.getGlobalTensorType(),
      xla::ifrt::IfrtShardingParamAttr::get(&ctx, sharding_param.value()),
      mesh_name_to_devices_attr.at(mesh_tensor_type.getMeshName()),
      mesh_tensor_type.getMemoryKind(), /*layout_attr=*/nullptr);
}

// Converts an MPMD TransferOp into an IFRT ReshardOp.
class TransferOpPattern final : public OpConversionPattern<TransferOp> {
 public:
  TransferOpPattern(const TypeConverter& type_converter, MLIRContext* context)
      : OpConversionPattern(type_converter, context) {}

  LogicalResult matchAndRewrite(
      TransferOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    const TypeConverter& type_converter = *getTypeConverter();
    Type converted_result_type = type_converter.convertType(op.getType());
    auto ifrt_reshard_op = xla::ifrt::ReshardOp::create(
        rewriter, op.getLoc(),
        /*outputs=*/converted_result_type,
        /*control_output=*/
        IfrtControlType::get(rewriter.getContext()),
        /*inputs=*/adaptor.getTensor(),
        /*donated=*/false,
        /*control_inputs=*/ValueRange());
    rewriter.replaceOp(op, ifrt_reshard_op.getOutputs());
    return success();
  }
};

// Converts an MPMD FragmentCallOp into an IFRT CallOp.
class FragmentCallOpPattern final : public OpConversionPattern<FragmentCallOp> {
 public:
  FragmentCallOpPattern(
      const TypeConverter& type_converter, MLIRContext* context,
      const llvm::StringMap<IfrtDevicesAttr>& mesh_name_to_devices_attr)
      : OpConversionPattern(type_converter, context),
        mesh_name_to_devices_attr_(mesh_name_to_devices_attr) {}

  LogicalResult matchAndRewrite(
      FragmentCallOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    const TypeConverter& type_converter = *getTypeConverter();
    // Convert the result types of the op.
    SmallVector<Type> converted_result_types;
    if (failed(type_converter.convertTypes(op->getResultTypes(),
                                           converted_result_types))) {
      return failure();
    }

    // Get the aliased inputs from the callee in order to construct the
    // io_aliases required by the ifrt.CallOp.
    std::vector<Attribute> io_aliases;
    std::vector<int> donated_input_indices;
    FuncOp callee =
        mlir::cast<FuncOp>(mlir::cast<CallOpInterface>(*op).resolveCallable());
    for (BlockArgument arg : callee.getArguments()) {
      if (auto donation_attr = callee.getArgAttrOfType<mlir::BoolAttr>(
              arg.getArgNumber(), kBufferDonationAttrName)) {
        donated_input_indices.push_back(arg.getArgNumber());
      }
      if (auto aliasing_attr = callee.getArgAttrOfType<mlir::IntegerAttr>(
              arg.getArgNumber(), kAliasingAttrName)) {
        io_aliases.push_back(rewriter.getDenseI32ArrayAttr(
            {static_cast<int32_t>(arg.getArgNumber()),
             static_cast<int32_t>(aliasing_attr.getInt())}));
      }
    }

    StringRef mesh_name = op.getMeshName();
    auto ifrt_call_op = xla::ifrt::CallOp::create(
        rewriter, op.getLoc(),
        /*outputs=*/converted_result_types,
        /*control_output=*/IfrtControlType::get(rewriter.getContext()),
        /*inputs=*/adaptor.getOperands(),
        /*control_inputs=*/ValueRange{},
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
    return success();
  }

 private:
  const llvm::StringMap<IfrtDevicesAttr>& mesh_name_to_devices_attr_;
};

// Pattern for converting the types of ReturnOps. It is used to ensure that the
// return of the main func is of type IFRT Array.
class ReturnOpPattern : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

// Updates the main function signature from MeshTensor to IFRT Array, and
// removes mesh shapes attributes from non-main functions.
class FuncOpPattern final : public OpConversionPattern<FuncOp> {
 public:
  FuncOpPattern(const TypeConverter& type_converter, MLIRContext* context)
      : OpConversionPattern(type_converter, context) {}

  LogicalResult matchAndRewrite(
      FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!IsMainFunction(op)) {
      // We only need to remove the mesh attributed for the non-main functions.
      rewriter.modifyOpInPlace(op, [&] { RemoveMesh(op); });
      return success();
    }
    FunctionType func_type = op.getFunctionType();
    // Convert the function signature.
    TypeConverter::SignatureConversion converted_args(func_type.getNumInputs());
    const TypeConverter& type_converter = *getTypeConverter();
    if (failed(type_converter.convertSignatureArgs(func_type.getInputs(),
                                                   converted_args))) {
      return failure();
    }
    // Convert the function results.
    SmallVector<Type> converted_results;
    if (failed(type_converter.convertTypes(func_type.getResults(),
                                           converted_results))) {
      return failure();
    }
    // Replace the types of the region arguments as per the signature result.
    if (failed(rewriter.convertRegionTypes(&op.getBody(), type_converter,
                                           &converted_args))) {
      return failure();
    }
    // Update the function signature.
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(FunctionType::get(rewriter.getContext(),
                                   converted_args.getConvertedTypes(),
                                   converted_results));
      op->removeAttr(mpmd::kTopologyAttr);
      op->setAttr(xla::ifrt::kIfrtFunctionAttrName, rewriter.getUnitAttr());
    });
    return success();
  }
};

// Replaces aliasing and donation attributes with the IFRT donated arg
// attribute. The attributes are removed here because they will be removed
// implicitly during stable serialization (i.e., conversion to VIFRT), which
// requires the attributes to be from the IFRT dialect. Thus, we replace them
// with `ifrt.donated` unit attribute, which is supported by VIFRT.
void ReplaceAliasingArgAttrsWithIfrtDonatedArgAttrs(FuncOp func) {
  // Mark donated args with the IFRT donated unit attribute.
  for (int arg_num = 0; arg_num < func.getNumArguments(); arg_num++) {
    if (func.getArgAttrOfType<mlir::BoolAttr>(arg_num,
                                              kBufferDonationAttrName)) {
      func.removeArgAttr(arg_num, kBufferDonationAttrName);
      func.setArgAttr(arg_num, xla::ifrt::kIfrtDonatedArgAttrName,
                      mlir::UnitAttr::get(func.getContext()));
    } else if (func.getArgAttrOfType<mlir::IntegerAttr>(arg_num,
                                                        kAliasingAttrName)) {
      func.removeArgAttr(arg_num, kAliasingAttrName);
      func.setArgAttr(arg_num, xla::ifrt::kIfrtDonatedArgAttrName,
                      mlir::UnitAttr::get(func.getContext()));
    }
  }
}

class LowerToIfrtPass
    : public PassWrapper<LowerToIfrtPass, mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToIfrtPass)

 private:
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<xla::ifrt::IfrtDialect>();
    xla::ifrt::AttachBuiltInSpmdExpansions(registry);
  }

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    MLIRContext& ctx = getContext();
    FuncOp main_func = GetMainFunction(module_op);

    ReplaceAliasingArgAttrsWithIfrtDonatedArgAttrs(main_func);

    // Construct mapping from mesh name to IFRT Device Attributes.
    llvm::StringMap<IfrtDevicesAttr> mesh_name_to_devices_attr;
    ArrayRef<mpmd::NamedMeshAttr> meshes = mpmd::GetTopologyMeshes(main_func);
    int total_devices = 0;
    for (const mpmd::NamedMeshAttr& mesh : meshes) {
      int num_mesh_devices = 1;
      for (sdy::MeshAxisAttr axis : mesh.getMesh().getAxes()) {
        num_mesh_devices *= axis.getSize();
      }
      SmallVector<int> mesh_devices(num_mesh_devices);
      absl::c_iota(mesh_devices, total_devices);
      total_devices += num_mesh_devices;
      mesh_name_to_devices_attr[mesh.getName()] =
          IfrtDevicesAttr::get(&ctx, mesh_devices);
    }

    mlir::ConversionTarget target(ctx);
    target.addIllegalDialect<MpmdDialect>();
    target.addLegalDialect<xla::ifrt::IfrtDialect,
                           mlir::stablehlo::StablehloDialect,
                           mlir::func::FuncDialect>();
    // The main func op should only have IFRT Array types, and the non-main
    // func ops should not have PartIR mesh_shape attribute.
    target.addDynamicallyLegalOp<FuncOp>(IsFuncOpLegal);
    target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
      return absl::c_all_of(op.getOperandTypes(), [](Type t) {
        return !mlir::isa<MeshTensorType>(t);
      });
    });

    // Set conversion from MeshTensorType to IFRT Array.
    TypeConverter type_converter;
    type_converter.addConversion([&meshes, &mesh_name_to_devices_attr](
                                     MeshTensorType mesh_tensor_type) -> Type {
      auto it = absl::c_find_if(
          meshes, [mesh_tensor_type](const mpmd::NamedMeshAttr& mesh) {
            return mesh.getName() == mesh_tensor_type.getMeshName();
          });
      CHECK(it != meshes.end())
          << "Mesh `" << mesh_tensor_type.getMeshName().str()
          << "` not found in topology.";
      sdy::MeshAttr mesh_attr = it->getMesh();
      return MeshTensorToArray(mesh_name_to_devices_attr, mesh_tensor_type,
                               mesh_attr);
    });
    // Do not convert if an Array is already an IFRT Array.
    type_converter.addConversion(
        [](xla::ifrt::IfrtArrayType ifrt_array_type) -> Type {
          return ifrt_array_type;
        });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuncOpPattern, TransferOpPattern>(type_converter, &ctx);
    patterns.add<ReturnOpPattern>(&ctx);
    patterns.add<FragmentCallOpPattern>(type_converter, &ctx,
                                        mesh_name_to_devices_attr);
    if (failed(mlir::applyPartialConversion(module_op, target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }

    // Convert the xla.sdy.meshes attribute to ifrt.sdy.meshes attribute so
    // that the attribute is preserved during IFRT versioning. This is safe
    // to do because the attribute if forward and backward compatible.
    if (auto front_end_attr = xla::sdy::getFrontendAttrs(module_op)) {
      if (auto meshes_round_trip_attr =
              front_end_attr.get(xla::sdy::kMeshesRoundTripAttr)) {
        module_op->setAttr(xla::ifrt::kIfrtSdyMeshesRoundTripAttr,
                           meshes_round_trip_attr);
      }
    }

    // Clean up the sdy meshes.
    mlir::IRRewriter rewriter(&ctx);
    auto sdy_mesh_op_s = module_op.getOps<sdy::MeshOp>();
    for (auto it = sdy_mesh_op_s.begin(); it != sdy_mesh_op_s.end();) {
      rewriter.eraseOp(*it++);
    }
  }

  StringRef getArgument() const override { return "mpmd-lower-to-ifrt"; }

  StringRef getDescription() const override {
    return "Lowers PartIR:MPMD to IFRT IR.";
  }
};

class AddCtrlDependenciesPass
    : public PassWrapper<AddCtrlDependenciesPass,
                         mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddCtrlDependenciesPass)

 private:
  void runOnOperation() override {
    FuncOp func_op = getOperation();
    // Mapping between a hash of devices used by the CallOp and the control
    // output of the last IFRT CallOp encountered that uses the same devices.
    DenseMap<llvm::ArrayRef<int>, TypedValue<IfrtControlType>>
        call_op_to_control_output;
    func_op.walk([&](xla::ifrt::CallOp call_op) {
      llvm::ArrayRef<int> devices = call_op.getDevices();
      if (TypedValue<IfrtControlType> ctrl_input =
              call_op_to_control_output.lookup(devices)) {
        call_op.getControlInputsMutable().append(ctrl_input);
      }
      call_op_to_control_output[devices] = call_op.getControlOutput();
      return WalkResult::skip();
    });
  }

  StringRef getArgument() const override {
    return "mpmd-ifrt-add-ctrl-dependencies";
  }

  StringRef getDescription() const override {
    return "Adds IFRT IR control dependencies to IFRT CallOps.";
  }
};

class BuildCompileOptionsPass
    : public PassWrapper<BuildCompileOptionsPass,
                         mlir::OperationPass<ModuleOp>> {
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
    ModuleOp module = getOperation();
    SymbolTableCollection symbol_table;
    FuncOp func_op = GetMainFunction(module);
    int threshold_for_argument_tupling = threshold_for_argument_tupling_;
    bool is_sdy_partitioned =
        module->hasAttr(xla::ifrt::kIfrtSdyMeshesRoundTripAttr);
    auto walk_result = func_op.walk([&](xla::ifrt::CallOp call_op) {
      xla::CompileOptions compile_options;
      xla::ExecutableBuildOptions& exec_build_options =
          compile_options.executable_build_options;
      ArrayRef<int> logical_device_ids = call_op.getDevicesAttr().getIds();
      exec_build_options.set_num_replicas(1);
      exec_build_options.set_num_partitions(logical_device_ids.size());
      xla::DeviceAssignment device_assignment(1, logical_device_ids.size());
      // Build options use IFRT logical device ids.
      for (const auto [i, device_id] : llvm::enumerate(logical_device_ids)) {
        device_assignment(0, i) = device_id;
      }
      exec_build_options.set_device_assignment(device_assignment);
      exec_build_options.set_use_spmd_partitioning(true);
      if (is_sdy_partitioned) {
        exec_build_options.set_use_shardy_partitioner(true);
      }
      FuncOp callee = call_op.getCalleeOp(symbol_table);
      if (auto reserved_hbm_bytes =
              callee->getAttrOfType<IntegerAttr>(kReservedHbmBytes)) {
        set_reserved_bytes_(exec_build_options, reserved_hbm_bytes.getInt());
      };
      std::string callee_name =
          callee->getParentOfType<mlir::ModuleOp>().getSymName()->str();
      auto mesh_name_attr =
          call_op->getAttrOfType<StringAttr>(kIfrtMeshNameAttrName);
      CHECK(mesh_name_attr != nullptr)
          << "ifrt.CallOp `" << callee_name << "` is missing "
          << kIfrtMeshNameAttrName.str() << " attribute.";
      // While the users provide per-mesh compilation options, we need to
      // include callee name in the key because fragments assigned to the same
      // mesh might have different `xla_tpu_user_reserved_hbm_bytes`.
      const std::string compile_options_key =
          absl::StrCat(callee_name, "_mesh_", mesh_name_attr.str());
      call_op->setAttr(
          xla::ifrt::kIfrtCompileOptionsKey,
          StringAttr::get(call_op->getContext(), compile_options_key));
      // Apply the user-provided per-mesh compile option overrides.
      if (auto option_overrides =
              compile_options_overrides_.find(mesh_name_attr.str());
          option_overrides != compile_options_overrides_.end()) {
        compile_options.env_option_overrides = option_overrides->second;
      }
      if (threshold_for_argument_tupling > 0 &&
          callee.getNumArguments() > threshold_for_argument_tupling) {
        compile_options.parameter_is_tupled_arguments = true;
      }
      compile_options_map_.emplace(compile_options_key, compile_options);
      return WalkResult::skip();
    });
    if (walk_result.wasInterrupted()) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "mpmd-ifrt-build-compile-options";
  }

  StringRef getDescription() const override {
    return "Gets the compile options for each IFRT atom program.";
  }
};

}  // namespace

std::unique_ptr<Pass> CreateLowerToIfrtPass() {
  return std::make_unique<LowerToIfrtPass>();
}

std::unique_ptr<Pass> CreateAddCtrlDependenciesPass() {
  return std::make_unique<AddCtrlDependenciesPass>();
}

std::unique_ptr<Pass> CreateBuildCompileOptionsPass(
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

void AddLowerToIfrtPasses(mlir::OpPassManager& pm,
                          bool add_control_dependencies) {
  pm.addPass(CreateLowerToIfrtPass());
  // IfrtMergeReshardsPass doesn't handle control dependencies, so we need to
  // run it before adding the control dependencies.
  pm.addNestedPass<mlir::func::FuncOp>(
      xla::ifrt::createIfrtMergeReshardsPass());
  if (add_control_dependencies) {
    pm.addNestedPass<FuncOp>(CreateAddCtrlDependenciesPass());
  }
  // Outline the IFRT atom programs to modules.
  xla::ifrt::IfrtToOutlinedAtomProgramsPipelineOptions outline_pipeline_options;
  outline_pipeline_options.propagate_shardings = false;
  xla::ifrt::createIfrtToOutlinedAtomProgramsPipeline(pm,
                                                      outline_pipeline_options);
}

void RegisterLowerToIfrtPasses() {
  mlir::registerPass(CreateLowerToIfrtPass);
  mlir::registerPass(xla::ifrt::createIfrtMergeReshardsPass);
  mlir::registerPass(CreateAddCtrlDependenciesPass);

  mlir::PassPipelineRegistration<> mpmd_lower_to_ifrt_pipeline(
      "ifrt-mpmd-lower-to-ifrt-pipeline", "Run the passes for lowering to ifrt",
      [](mlir::OpPassManager& pm) {
        AddLowerToIfrtPasses(pm, /*add_control_dependencies=*/true);
      });
}

absl::Status LowerToIfrt(mlir::ModuleOp module, bool add_control_dependencies) {
  FuncOp main_func = GetMainFunction(module);
  if (!mpmd::IsMpmdFunction(main_func)) {
    return absl::InvalidArgumentError("MLIR module is not an MPMD module.");
  }
  mlir::PassManager pm(module->getContext());
  InitPassManager(pm, "mpmd-lower-to-ifrt");
  pm.enableVerifier();

  // If we are lowered with SDY, we need to run the SDY round trip pipeline.
  if (mlir::mpmd::IsLoweredWithSdy(module)) {
    xla::sdy::addSdyRoundTripExportPipeline(pm);
  }
  AddLowerToIfrtPasses(pm, add_control_dependencies);
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
  FuncOp main_func = GetMainFunction(module);
  if (!IsIfrtFunction(main_func)) {
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
