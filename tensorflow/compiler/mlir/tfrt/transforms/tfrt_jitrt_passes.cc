/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_clustering.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/fallback_converter.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_jitrt_stub.h"
#include "tfrt/jitrt/opdefs/jitrt_ops.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime

namespace tensorflow {
namespace {

class TfrtJitRtStubImpl : public TfrtJitRtStub {
  void RegisterJitRtDialects(mlir::DialectRegistry &registry) override;

  void PopulateJitRtConversionPatterns(
      mlir::ConversionTarget *target, mlir::MLIRContext *context,
      mlir::RewritePatternSet *patterns,
      CoreRTConverter *corert_converter) override;

  mlir::Value CreateJitRtFallbackCompileKernel(
      mlir::OpBuilder &builder, mlir::ModuleOp module,
      mlir::Value chain_value) override;

  void AddTfrtJitRtPasses(const TfrtPipelineOptions &options,
                          mlir::OpPassManager &pm) override;
};

void TfrtJitRtStubImpl::RegisterJitRtDialects(mlir::DialectRegistry &registry) {
  registry.insert<tf_jitrt::JitRuntimeDialect>();
}

// TODO(ezhulenev): tf_device.cluster operations after auto-fusion should
// have the correct device assigned based on the fused operations. We should
// use this device to convert operands and results from/to corert handles.
// For now it is safe to assume that it is "CPU" because we do not support
// any other devices and do not support distributed models.
constexpr char kJitRtDevice[] = "/job:localhost/replica:0/task:0/device:CPU:0";

// Convert jitrt.call operations to the tf_jitrt.fallback.execute operation.
class JitRtCallToJitRtCompileAndExecuteConversion
    : public OpConversionPattern<tfrt::jitrt::CallOp> {
 public:
  explicit JitRtCallToJitRtCompileAndExecuteConversion(MLIRContext *context)
      : OpConversionPattern<tfrt::jitrt::CallOp>(context) {}

  LogicalResult matchAndRewrite(
      tfrt::jitrt::CallOp call, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Convert operands to fallback tensors.
    llvm::SmallVector<Value, 4> fallback_operands;
    if (failed(tfrt_compiler::ConvertFallbackOperands(
            call, kJitRtDevice, adaptor.getOperands(), &fallback_operands,
            rewriter)))
      return rewriter.notifyMatchFailure(call, "failed to convert operand");

    // tf_jitrt.fallback.execute always produces fallback tensors.
    llvm::SmallVector<Type, 4> result_types(
        call->getNumResults(),
        rewriter.getType<tfrt::fallback::TFTensorType>());

    // Replace jitrt.call operation with a tf_jitrt.fallback.execute operation.
    rewriter.replaceOpWithNewOp<tf_jitrt::FallbackExecuteOp>(
        call, result_types, call.getCallee(), fallback_operands, kJitRtDevice);

    return success();
  }
};

// Helper function for inserting TFRT JitRt dialect conversions.
void TfrtJitRtStubImpl::PopulateJitRtConversionPatterns(
    mlir::ConversionTarget *target, MLIRContext *context,
    RewritePatternSet *patterns, CoreRTConverter *corert_converter) {
  target->addLegalDialect<tf_jitrt::JitRuntimeDialect>();
  target->addIllegalDialect<tfrt::jitrt::JitRuntimeDialect>();
  // Lower jitrt.call to the pair of compile and execute operations.
  patterns->add<JitRtCallToJitRtCompileAndExecuteConversion>(context);
}

mlir::Value TfrtJitRtStubImpl::CreateJitRtFallbackCompileKernel(
    mlir::OpBuilder &builder, mlir::ModuleOp module, mlir::Value chain_value) {
  // Pre-compile all JIT compiled kernels found in the module.
  llvm::SmallVector<Value> compiled;

  // A set SymbolRef attributes referencing compiled kernels.
  llvm::DenseSet<mlir::Attribute> kernels;

  // Compile all kernels in parallell.
  module.walk([&](tf_jitrt::FallbackExecuteOp execute) {
    // Do not compiled the same kernel multiple times.
    if (kernels.contains(execute.getKernel())) return;

    auto compile = builder.create<tf_jitrt::FallbackCompileOp>(
        execute.getLoc(), builder.getType<tfrt::compiler::ChainType>(),
        execute.getKernel(), execute.getDevice());
    compiled.push_back(compile.getResult());
    kernels.insert(compile.getKernel());
  });

  // Wait for the compilation completion before returning from init function.
  if (!compiled.empty()) {
    // Do not forget to wait for the fallback kernels initialization.
    compiled.insert(compiled.begin(), chain_value);
    chain_value = builder.create<tfrt::compiler::MergeChainsOp>(
        module.getLoc(), builder.getType<tfrt::compiler::ChainType>(),
        compiled);
  }

  return chain_value;
}

// -------------------------------------------------------------------------- //
// Outline tf_device.cluster operation regions into functions in the nested
// modules and replaces all cluster operations with jitrt.call operations.
// -------------------------------------------------------------------------- //

class OutlineJitRtClustersPass
    : public PassWrapper<OutlineJitRtClustersPass, OperationPass<ModuleOp>> {
 public:
  llvm::StringRef getArgument() const final {
    return "tf-outline-jitrt-cluster";
  }
  llvm::StringRef getDescription() const final {
    return "Outlines `tf_device.cluster` operations into functions and "
           "replaces them with `jitrt.call` operations.";
  }

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<tfrt::jitrt::JitRuntimeDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineJitRtClustersPass)

 private:
  struct CompiledModule {
    ModuleOp module;
    func::FuncOp entrypoint;
    llvm::SetVector<Value> operands;
  };

  // Creates a nested module with a single function that will be compiled into
  // the kernel at runtime.
  CompiledModule CreateCompiledModule(tf_device::ClusterOp cluster,
                                      int64_t max_arg_size,
                                      SymbolTable *symbol_table);

  // Update compiled module entrypoint signature with inferred operands
  // constraints.
  LogicalResult SetEntrypointConstraints(CompiledModule &compiled);

  // Outlines cluster operation regions into compiled modules, and replaces
  // cluster operation with a jitrt.call operation.
  LogicalResult OutlineClusterOp(tf_device::ClusterOp cluster,
                                 int64_t max_arg_size,
                                 SymbolTable *symbol_table);

  // Mapping from the outlined module string representation to the module itself
  // and an entrypoint function. Used to deduplicate identical modules during
  // the `tf_device.cluster` outlining.
  llvm::StringMap<std::pair<ModuleOp, func::FuncOp>> outlined_;
};

OutlineJitRtClustersPass::CompiledModule
OutlineJitRtClustersPass::CreateCompiledModule(tf_device::ClusterOp cluster,
                                               int64_t max_arg_size,
                                               SymbolTable *symbol_table) {
  MLIRContext *ctx = cluster->getContext();
  Location loc = cluster.getLoc();

  // Create a module that will hold compiled function and async wrappers.
  // TODO(ezhulenev): Give better names to module and function.
  auto compiled_module = ModuleOp::create(loc, {"kernel"});
  compiled_module->setAttr("tfrt.compiled", UnitAttr::get(ctx));
  compiled_module->setAttr(
      "tfrt.max-arg-size",
      IntegerAttr::get(IntegerType::get(ctx, 64), max_arg_size));

  SymbolTable compiled_module_symbol_table(compiled_module);

  // Find out the cluster arguments and their types.
  llvm::SetVector<Value> live_ins;
  getUsedValuesDefinedAbove(cluster.getBody(), cluster.getBody(), live_ins);

  llvm::SmallVector<Type, 4> operand_types;
  operand_types.reserve(live_ins.size());
  for (Value v : live_ins) operand_types.emplace_back(v.getType());

  // Create a function in the compiled module.
  auto compiled_func_type =
      FunctionType::get(ctx, operand_types, cluster->getResultTypes());
  auto compiled_func = func::FuncOp::create(loc, "compute", compiled_func_type);
  compiled_module_symbol_table.insert(compiled_func);

  // Replace uses of live-in values within cluster region with block arguments.
  Block *compiled_func_block = compiled_func.addEntryBlock();
  for (auto p : llvm::zip(live_ins, compiled_func_block->getArguments()))
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                               cluster.getBody());

  // Move all operations in cluster into compiled_func's entry block.
  auto &cluster_body = cluster.GetBody().getOperations();
  compiled_func_block->getOperations().splice(
      compiled_func_block->end(), cluster_body, cluster_body.begin(),
      cluster_body.end());

  // Replace `tf_device.return` terminator with `func.return` in the function
  // body.
  auto device_return =
      cast<tf_device::ReturnOp>(compiled_func_block->getTerminator());
  OpBuilder builder(device_return.getOperation());
  builder.create<func::ReturnOp>(device_return.getLoc(),
                                 device_return.getOperands());
  device_return.erase();

  // TODO(ezhulenev): MLIR doesn't define operation equivalence upstream yet,
  // replace module printing with a more principled solution when available.
  // Operations in the cluster can be in different order, however define the
  // identical Tensorflow programs, with current approach we'll not be able
  // to detect duplicates like this.

  // Remove location attribute attached to Tensorflow operations to be able to
  // deduplicate compiled clusters with the same set of operations.
  //
  // TODO(ezhulenev): Figure out how to propagate locations for error reporting,
  // right now JitRt will ignore them anyway.
  compiled_module.walk([](Operation *op) { op->removeAttr("_class"); });

  // Serialize prepared module to string.
  std::string serialized;
  llvm::raw_string_ostream os(serialized);
  compiled_module.print(os);

  // Try to find if identical module was already outlined.
  auto it = outlined_.find(serialized);

  // Return identical module that was already outlined earlier.
  if (it != outlined_.end()) {
    compiled_module.erase();  // erase identical module
    return {it->second.first, it->second.second, live_ins};
  }

  // Insert compiled module into the symbol table and assign it a unique name.
  symbol_table->insert(compiled_module);

  // Cache unique module.
  outlined_.insert({std::move(serialized), {compiled_module, compiled_func}});

  return {compiled_module, compiled_func, live_ins};
}

LogicalResult OutlineJitRtClustersPass::SetEntrypointConstraints(
    CompiledModule &compiled) {
  func::FuncOp func = compiled.entrypoint;

  // Functions outlined from jitrt device clusters must have a single block.
  assert(func.getBody().getBlocks().size() == 1 && "expected single block");

  mlir::TFDevice::ClusteringPolicySet policies;
  populateTfJitRtConstraintsPolicies(policies);

  // Infer constraints on the values defined in the entrypoint function
  // (including function entry block arguments).
  mlir::TFDevice::ValuesConstraintSet constraints;
  if (failed(mlir::TFDevice::PropagateValuesConstraints(
          func.getBody(), policies, constraints, /*resolve=*/true)))
    return failure();

  // Annotate arguments with inferred constraints.
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    if (auto constraint = constraints.GetConstraint(func.getArgument(i))) {
      auto constraint_name = mlir::StringAttr::get(
          &getContext(), llvm::formatv("{0}", *constraint).str());
      func.setArgAttr(i, "rt.constraint", constraint_name);
    }
  }

  return success();
}

LogicalResult OutlineJitRtClustersPass::OutlineClusterOp(
    tf_device::ClusterOp cluster, int64_t max_arg_size,
    SymbolTable *symbol_table) {
  Location loc = cluster->getLoc();
  OpBuilder builder(cluster);

  CompiledModule compiled_module =
      CreateCompiledModule(cluster, max_arg_size, symbol_table);
  func::FuncOp compiled_func = compiled_module.entrypoint;

  // Add constraints to the entrypoint arguments.
  if (failed(SetEntrypointConstraints(compiled_module))) return failure();

  // Replace device cluster with a jitrt.call operation.
  auto module_name = *compiled_module.module.getSymName();
  auto func_name = compiled_func.getSymName();
  auto func_flat_ref =
      mlir::SymbolRefAttr::get(builder.getContext(), func_name);
  auto func_ref = mlir::SymbolRefAttr::get(builder.getContext(), module_name,
                                           {func_flat_ref});

  auto cluster_func_op = builder.create<tfrt::jitrt::CallOp>(
      loc, cluster.getResultTypes(), func_ref,
      compiled_module.operands.getArrayRef());

  cluster.replaceAllUsesWith(cluster_func_op);
  cluster.erase();

  return success();
}

void OutlineJitRtClustersPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);

  // Keep track of the maximum argument size for each function with tf_device
  // cluster operations in the function body. We need to pass it to the compiled
  // module to correctly compute its cost later.
  llvm::DenseMap<mlir::func::FuncOp, int64_t> max_arg_size_map;

  auto get_max_arg_size = [&](mlir::func::FuncOp func) -> int64_t {
    auto it = max_arg_size_map.find(func);
    if (it != max_arg_size_map.end()) return it->second;
    return max_arg_size_map[func] = tf_jitrt::GetMaxArgSize(func);
  };

  OpBuilder builder(module.getContext());
  auto result = module.walk([&](tf_device::ClusterOp cluster) -> WalkResult {
    // Ensure that cluster was formed for TFRT JIT compilation.
    auto policy = cluster->getAttr("policy").dyn_cast_or_null<StringAttr>();
    if (!policy || policy.getValue() != "tfrt.auto-fusion")
      return WalkResult::advance();

    // Get the maximum argument size of the parent function.
    mlir::func::FuncOp parent_func =
        cluster->getParentOfType<mlir::func::FuncOp>();
    int64_t max_arg_size = get_max_arg_size(parent_func);

    if (failed(OutlineClusterOp(cluster, max_arg_size, &symbol_table)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    module->emitError("Failed to outline tf_device.cluster operations");
    signalPassFailure();
  }
}

std::unique_ptr<Pass> CreateOutlineJitRtClustersPass() {
  return std::make_unique<OutlineJitRtClustersPass>();
}

void TfrtJitRtStubImpl::AddTfrtJitRtPasses(const TfrtPipelineOptions &options,
                                           mlir::OpPassManager &pm) {
  // Outline auto-fusion clusters into tf_device.cluster_operations and then
  // convert them to functions. We currently support only tfrt fallback tensors
  // as operands, so we disable these passes if we can have native ops after
  // lowering.
  pm.addNestedPass<mlir::func::FuncOp>(CreateTfJitRtClusteringPass(
      options.auto_fusion_oplist, options.auto_fusion_min_cluster_size));

  // Sink small constants into the outlined clusters to reduce the number of
  // arguments for each of the execute operations.
  auto is_compilable_const = [](mlir::tf_device::ClusterOp cluster,
                                mlir::ElementsAttr value) -> bool {
    // Ensure that cluster was formed for TFRT JIT compilation.
    auto policy = cluster->getAttr("policy").dyn_cast_or_null<StringAttr>();
    if (!policy || policy.getValue() != "tfrt.auto-fusion") return false;

    // Check that TF->JitRt compiler supports constant compilation.
    return mlir::succeeded(IsCompilableConstant(value));
  };

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateClusterConstantSinkingPass(is_compilable_const));

  // Outline formed JIT compiled device clusters into function.
  pm.addPass(CreateOutlineJitRtClustersPass());
}

mlir::PassRegistration<OutlineJitRtClustersPass> tf_outline_jitrt_cluster_pass(
    CreateOutlineJitRtClustersPass);

const bool kUnused =
    (RegisterTfrtJitRtStub(std::make_unique<TfrtJitRtStubImpl>()), true);

}  // namespace
}  // namespace tensorflow
