/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_dataflow.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

namespace mlir {
namespace tf_executor {
namespace {

using TF::ResourceId;
using ResourceAndDevice = std::pair<ResourceId, int32_t>;
static constexpr ResourceId kUnknownResourceId =
    TF::detail::ResourceAliasAnalysisInfo::kUnknownResourceId;
static constexpr ResourceId kInvalidResourceId =
    TF::detail::ResourceAliasAnalysisInfo::kInvalidResourceId;
using OperationSetTy = SmallPtrSet<Operation*, 4>;
using ResourceToOpsMapTy = DenseMap<ResourceAndDevice, OperationSetTy>;
using DeviceMap = DenseMap<StringAttr, int64_t>;
constexpr int64_t kAnyDevice = 0;
constexpr ResourceAndDevice kInvalidResourceAndDevice{kInvalidResourceId,
                                                      kAnyDevice};
constexpr ResourceAndDevice kUnknownResourceAndDevice{kUnknownResourceId,
                                                      kAnyDevice};

constexpr char kDeviceAttr[] = "device";

#define GEN_PASS_DEF_EXECUTORCONVERTCONTROLTODATAOUTPUTSPASS
#define GEN_PASS_DECL_EXECUTORCONVERTCONTROLTODATAOUTPUTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ConvertControlToDataOutputsPass
    : public impl::ExecutorConvertControlToDataOutputsPassBase<
          ConvertControlToDataOutputsPass> {
  ConvertControlToDataOutputsPass() = default;
  explicit ConvertControlToDataOutputsPass(
      bool composite_tpuexecute_side_effects)
      : ExecutorConvertControlToDataOutputsPassBase(
            ExecutorConvertControlToDataOutputsPassOptions{
                composite_tpuexecute_side_effects}) {}

  void runOnOperation() override;
};

// Returns a vector of all tf.WhileOp(s) which use func as while body. If any of
// the uses is as a while condition, an empty vector is returned.
SmallVector<TF::WhileOp> GetWhileCallers(func::FuncOp func,
                                         SymbolUserMap& symbol_map) {
  SmallVector<TF::WhileOp> while_callers;
  for (auto user : symbol_map.getUsers(func)) {
    if (auto while_caller = dyn_cast<TF::WhileOp>(user)) {
      // If used as while conditional anywhere, then skip optimizing this
      // function. Return empty vector.
      if (while_caller.cond_function() == func) return {};
      assert(while_caller.body_function() == func);
      while_callers.push_back(while_caller);
    }
  }
  return while_callers;
}

bool IsResourceType(Type type) {
  return mlir::isa<TF::ResourceType>(getElementTypeOrSelf(type));
}

bool OnlyOperatesOnCompositeDevices(
    TF::TPUExecuteAndUpdateVariablesOp& op,
    const TF::SideEffectAnalysis::Info& side_effect_analysis,
    const DataFlowSolver& solver) {
  auto& alias_analysis = side_effect_analysis.GetAliasAnalysis();
  llvm::SmallSet<int, 8> read_array;
  for (const Attribute& attr : op.getDeviceVarReadsIndices()) {
    read_array.insert(mlir::cast<IntegerAttr>(attr).getInt());
  }
  llvm::SmallSet<int, 8> update_array;
  for (const Attribute& attr : op.getDeviceVarUpdatesIndices()) {
    update_array.insert(mlir::cast<IntegerAttr>(attr).getInt());
  }

  for (auto& arg : op->getOpOperands()) {
    Value v = arg.get();
    if (!IsResourceType(arg.get().getType())) continue;
    if (alias_analysis.IsUnknownResource(v)) continue;
    for (auto id : alias_analysis.GetResourceUniqueIds(v)) {
      (void)id;
    }
  }

  for (auto& arg : op->getOpOperands()) {
    if (!IsResourceType(arg.get().getType())) {
      continue;
    }
    auto lattice =
        solver.lookupState<TF::IsCompositeDataflowState>(arg.get())->getValue();
    bool is_read = read_array.contains(arg.getOperandNumber());
    bool is_update = update_array.contains(arg.getOperandNumber());
    // We want the resource operands that are on composite devices to be the
    // exact same set as the resource operands that are read or updated.
    if ((is_read || is_update) != lattice.is_on_composite_device) {
      return false;
    }
  }
  return true;
}

// Populates `chain_resource_to_ops_map`, the map from all resources that need
// to be chained to the set of operations that access the resource, and
// `resource_equivalence_classes`. Resources are equivalent if they are accessed
// by a common op, and equivalent resources will be assigned to the same chain.
void CollectChainResources(
    func::FuncOp func, ResourceToOpsMapTy& chain_resource_to_ops_map,
    llvm::EquivalenceClasses<ResourceAndDevice>& resource_equivalence_classes,
    DeviceMap& devices,
    const TF::SideEffectAnalysis::Info& side_effect_analysis,
    const DataFlowSolver& solver, bool composite_tpuexecute_side_effects) {
  auto graph_op = cast<GraphOp>(func.front().front());

  // For each op in the graph, get the resources it uses and update the access
  // information for them.
  graph_op.walk([&](IslandOp island) {
    // This pass assumes that all functions are suitable for export i.e., each
    // function has a single tf_executor.graph op and all islands wrap the
    // internal op perfectly. Hence this assertion should never fail.
    assert(island.WrapsSingleOp());
    Operation& op = island.GetBody().front();

    // If the op only operates on resources stored on devices that are
    // "COMPOSITE", then this op is defined to work in parallel with other
    // TPUExecute* ops. So we can make all ResourceIds device-specific below.
    // (Even the per-op "resource ids", like ResourceEffects::TPUExecute.)
    bool op_only_operates_on_composite_devices = false;
    if (auto execute = llvm::dyn_cast<TF::TPUExecuteAndUpdateVariablesOp>(op)) {
      if (OnlyOperatesOnCompositeDevices(execute, side_effect_analysis,
                                         solver)) {
        op_only_operates_on_composite_devices = true;
      }
    }

    auto device_attr = op.getAttrOfType<StringAttr>(kDeviceAttr);
    int64_t device_id;
    if (!device_attr) {
      device_id = kAnyDevice;
    } else if (devices.find(device_attr) != devices.end()) {
      device_id = devices[device_attr];
    } else {
      device_id = 1 + devices.size();
      devices[device_attr] = device_id;
    }

    auto& alias_analysis = side_effect_analysis.GetAliasAnalysis();

    ResourceAndDevice prev_resource_and_device = kInvalidResourceAndDevice;
    for (auto resource_id_read_only_pair :
         side_effect_analysis.GetResourceIds(&op)) {
      auto resource_id = resource_id_read_only_pair.first;
      // If alias analysis knows about a resource (as evidenced by the fact that
      // GetValuesForResourceId isn't empty), and dataflow tells us that this
      // stems from a function argument that was annotated as
      // "tf._composite_device", then we can treat this resource as
      // device-specific (see below).
      bool resource_is_on_composite_device = false;
      for (Value value : alias_analysis.GetValuesForResourceId(resource_id)) {
        auto lattice = solver.lookupState<TF::IsCompositeDataflowState>(value);
        if (lattice) {
          resource_is_on_composite_device |=
              lattice->getValue().is_on_composite_device;
        }
      }

      // A device-specific resource identifier creates an edge only between ops
      // on the same device, thus preventing ops on different devices from
      // blocking each other, even if they access the same resource.
      ResourceAndDevice resource_and_device;
      if (composite_tpuexecute_side_effects &&
          (op_only_operates_on_composite_devices ||
           resource_is_on_composite_device)) {
        resource_and_device = std::make_pair(resource_id, device_id);
      } else {
        resource_and_device = std::make_pair(resource_id, kAnyDevice);
      }

      // If the resource was allocated by an op with `UniqueResourceAllocation`
      // trait, then we don't need to chain resource ops accessing this resource
      // between iterations: Every iteration will create a new independent
      // resource. This enables more parallelism across iterations.
      if (!side_effect_analysis.IsUniqueResourceAllocationId(
              resource_and_device.first)) {
        chain_resource_to_ops_map[resource_and_device].insert(&op);
        if (prev_resource_and_device != kInvalidResourceAndDevice) {
          // Merge class of current ID with class of previous ID since both
          // resources are accessed by `op`.
          resource_equivalence_classes.unionSets(prev_resource_and_device,
                                                 resource_and_device);
        } else {
          resource_equivalence_classes.insert(resource_and_device);
        }
        prev_resource_and_device = resource_and_device;
      }
    }
  });
}

// tf.NoOp islands are used to combine multiple control dependencies into one.
// These islands have a single tf.NoOp inside them and consume multiple control
// outputs to generate a single control output.
//
// For example,
// ```
// %merged_control = "tf_executor.island"(%control_a, %control_b) ({
//   "tf.NoOp"() : () -> ()
//   "tf_executor.yield"() : () -> ()
// }) : (!tf_executor.control, !tf_executor.control) -> (!tf_executor.control)
// ```
//
// `%merged_control` is a NoOp control barrier in this case.
//
// Checks if the value `control` is a NoOp control barrier.
bool IsNoOpControlBarrier(Value control) {
  if (!mlir::isa<ControlType>(control.getType())) return false;

  auto control_island = dyn_cast_or_null<IslandOp>(control.getDefiningOp());
  if (!control_island) return false;

  // All islands perfectly wrap a single op is an invariant of this pass and
  // is checked at the very beginning of the pass.
  assert(control_island.WrapsSingleOp());
  return control_island.getOutputs().empty() &&
         isa<TF::NoOp>(control_island.GetBody().front());
}

// Remove all control outputs of the function. Traverses NoOp control barrier
// chains from FetchOp to all NoOp control barriers. Returns true
// iff at least one control output is deleted. The ops_connected_to_fetch
// set is populated with all operations that had a direct (or indirect, through
// Identity ops) control connection to the fetch. That set will contain nullptr
// if an arg is connected to the fetch.
bool RemoveAllControlOutputs(
    func::FuncOp func, SmallPtrSet<Operation*, 4>* ops_connected_to_fetch) {
  auto graph_op = cast<GraphOp>(func.front().front());

  FetchOp fetch = graph_op.GetFetch();
  // Return early if no control outputs exist.
  if (fetch.getNumOperands() == graph_op->getNumResults()) return false;

  std::queue<Value> control_barrier_worklist;
  for (Value control_input :
       fetch.getFetches().drop_front(graph_op->getNumResults())) {
    ops_connected_to_fetch->insert(control_input.getDefiningOp());
    if (IsNoOpControlBarrier(control_input))
      control_barrier_worklist.push(control_input);
  }

  // Erase all control outputs at the end from fetch.
  fetch.getFetchesMutable().erase(
      graph_op.getNumResults(),
      fetch.getNumOperands() - graph_op.getNumResults());

  // Iterate the worklist to remove all NoOp control barriers at the end of the
  // function body that are used to merge two or more control dependencies.
  while (!control_barrier_worklist.empty()) {
    Value control_barrier = control_barrier_worklist.front();
    control_barrier_worklist.pop();

    // We can only erase control barriers whose uses have been erased as well.
    if (!control_barrier.use_empty()) continue;

    // Only values defined by IslandOp were inserted in the worklist.
    IslandOp current_island = cast<IslandOp>(control_barrier.getDefiningOp());

    for (auto control_input : current_island.getControlInputs()) {
      ops_connected_to_fetch->insert(control_input.getDefiningOp());
      if (IsNoOpControlBarrier(control_input))
        control_barrier_worklist.push(control_input);
    }
    current_island.erase();
    ops_connected_to_fetch->erase(current_island);
  }
  return true;
}

// Appends function arguments with `num_resources` number of arguments of
// requested type.
void AppendFunctionArguments(func::FuncOp func, int num_resources,
                             ShapedType chaining_data_type) {
  for (int i = 0; i < num_resources; ++i) {
    func.getRegion().addArgument(chaining_data_type, func.getLoc());
  }

  FunctionType ftype =
      FunctionType::get(func.getContext(), func.getBody().getArgumentTypes(),
                        func.getFunctionType().getResults());
  func.setType(ftype);
}

// Appends function results with `num_resources` number of results of requested
// type.
void AppendFunctionResults(func::FuncOp func, int num_resources,
                           ShapedType chaining_data_type) {
  Block& block = func.front();
  auto graph_op = cast<GraphOp>(block.front());
  // Note that func result types are same as the result types of
  // GraphOp in the function `func`.
  assert(std::equal(func->getResultTypes().begin(),
                    func->getResultTypes().end(),
                    graph_op->getResultTypes().begin()));
  auto new_result_types =
      llvm::to_vector<4>(func.getFunctionType().getResults());
  for (int i = 0; i < num_resources; ++i) {
    new_result_types.push_back(chaining_data_type);
  }
  FunctionType ftype = FunctionType::get(
      func.getContext(), func.getArgumentTypes(), new_result_types);
  func.setType(ftype);

  // Rewrite GraphOp to have same number of results as the
  // function.
  OpBuilder builder(graph_op);
  auto new_graph_op =
      builder.create<GraphOp>(graph_op.getLoc(), new_result_types);
  new_graph_op.getRegion().takeBody(graph_op.getRegion());
  graph_op->replaceAllUsesWith(
      new_graph_op->getResults().drop_back(num_resources));
  graph_op.erase();
  func::ReturnOp return_op = cast<func::ReturnOp>(block.getTerminator());
  int num_old_arguments = return_op.getNumOperands();
  return_op->insertOperands(
      num_old_arguments,
      new_graph_op.getResults().slice(num_old_arguments, num_resources));
}

// Creates a wrapper island enclosing the `sub_op` dependent on
// `control_inputs`.
IslandOp CreateIsland(Operation* sub_op, ValueRange control_inputs,
                      OpBuilder builder) {
  assert(sub_op);
  auto control_type = ControlType::get(builder.getContext());
  auto island = builder.create<IslandOp>(
      sub_op->getLoc(), sub_op->getResultTypes(), control_type, control_inputs);
  island.getBody().push_back(new Block);
  Block* block = &island.getBody().back();
  builder.setInsertionPointToEnd(block);
  sub_op->replaceAllUsesWith(island.getOutputs());
  sub_op->moveBefore(block, block->begin());
  builder.create<YieldOp>(sub_op->getLoc(), sub_op->getResults());
  return island;
}

// Adds control dependencies from/to chain arguments/results. It adds two
// identity ops, chain_src and chain_sink, per resource equivalence class.
// Using the resource to operations map, it adds (1) a control dependency
// from chain_src to all the operations that read/write to a resource of the
// equivalence class, and (2) a control dependency from all the operations that
// read/write to a resource of the class to the chain_sink operation.
void ChainResourceOps(
    func::FuncOp func, ResourceToOpsMapTy& chain_resource_to_ops_map,
    llvm::EquivalenceClasses<ResourceAndDevice>& resource_equivalence_classes,
    SmallPtrSet<Operation*, 4> ops_connected_to_fetch, int num_old_outputs) {
  assert(num_old_outputs + resource_equivalence_classes.getNumClasses() ==
         func.getNumArguments());
  auto graph_op = cast<GraphOp>(func.front().front());

  auto fetch = graph_op.GetFetch();
  OpBuilder builder_chain_src(fetch);
  builder_chain_src.setInsertionPointToStart(fetch->getBlock());

  OpBuilder builder_chain_sink(fetch);
  int chain_index = num_old_outputs;

  // Iterate over all equivalence classes.
  for (auto class_iter = resource_equivalence_classes.begin();
       class_iter != resource_equivalence_classes.end(); ++class_iter) {
    // Only visit one element per class, the leader.
    if (!class_iter->isLeader()) continue;

    // Create chain source and sink identity islands for current equivalence
    // class.
    auto chain_arg = func.getArgument(chain_index++);
    auto src_identity = builder_chain_src.create<TF::IdentityOp>(
        chain_arg.getLoc(), chain_arg.getType(), chain_arg);
    auto chain_src_island = CreateIsland(src_identity, {}, builder_chain_src);

    auto sink_identity = builder_chain_sink.create<TF::IdentityOp>(
        chain_arg.getLoc(), chain_arg.getType(), chain_arg);
    auto chain_sink_island =
        CreateIsland(sink_identity, {}, builder_chain_sink);

    // Add the chain sink data output to fetch. These might stay empty.
    fetch.getFetchesMutable().append(chain_sink_island.getOutputs().front());

    // Iterate over all members of the current equivalence class (represented
    // by `class_iter`). Keep track of ops that have already been processed.
    llvm::SmallDenseSet<Operation*> processed_ops;
    for (auto member_iter =
             resource_equivalence_classes.member_begin(class_iter);
         member_iter != resource_equivalence_classes.member_end();
         ++member_iter) {
      ResourceAndDevice resource_and_device = *member_iter;
      auto map_iter = chain_resource_to_ops_map.find(resource_and_device);
      if (map_iter == chain_resource_to_ops_map.end()) continue;
      OperationSetTy& resource_ops = map_iter->getSecond();

      // Add dependencies between all ops that access current resource and chain
      // source and sink.
      for (Operation* op : resource_ops) {
        if (processed_ops.contains(op)) continue;

        IslandOp wrapper = op->getParentOfType<IslandOp>();
        assert(wrapper);
        wrapper.getControlInputsMutable().append(chain_src_island.getControl());
        if (ops_connected_to_fetch.contains(wrapper)) {
          chain_sink_island.getControlInputsMutable().append(
              wrapper.getControl());
        }
        processed_ops.insert(op);
      }
    }
  }
  VLOG(2) << "Added " << resource_equivalence_classes.getNumClasses()
          << " chains for " << chain_resource_to_ops_map.size() << " resources";
}

// Generate a dummy constant island of requested type.
IslandOp GetDummyConstant(OpBuilder builder, ShapedType const_type,
                          Location loc) {
  DenseIntElementsAttr val = DenseIntElementsAttr::get(const_type, 1);
  auto const_op = builder.create<TF::ConstOp>(loc, val);
  auto const_island = CreateIsland(const_op, {}, builder);
  return const_island;
}

// Rewrites the while op with extra chaining operands and results. Uses a
// dummy constant of requested type as argument to all the new chaining
// operands.
TF::WhileOp RewriteWhileOp(TF::WhileOp while_op, int num_resource_inputs,
                           ShapedType const_type) {
  IslandOp while_wrapper = while_op->getParentOfType<IslandOp>();
  assert(while_wrapper && "While op is expected to be wrapped in a IslandOp");

  // Get the dummy constant.
  OpBuilder builder(while_wrapper);
  auto loc = NameLoc::get(
      builder.getStringAttr("chain_control_outputs@" + while_op.getBody()));
  IslandOp const_wrapper = GetDummyConstant(builder, const_type, loc);

  // Get new operand and result types.
  auto new_operands = llvm::to_vector<4>(while_op->getOperands());
  auto new_result_types = llvm::to_vector<4>(while_op->getResultTypes());
  Value const_output = const_wrapper.getOutputs()[0];
  for (int i = 0; i < num_resource_inputs; ++i) {
    new_operands.push_back(const_output);
    new_result_types.push_back(const_output.getType());
  }

  // Replace old while op with new while op.
  auto new_while_op = builder.create<TF::WhileOp>(
      while_op.getLoc(), new_result_types, new_operands, while_op->getAttrs());
  auto new_while_wrapper =
      CreateIsland(new_while_op, while_wrapper.getControlInputs(), builder);
  for (auto result : while_wrapper.getOutputs()) {
    result.replaceAllUsesWith(
        new_while_wrapper.getOutputs()[result.getResultNumber()]);
  }
  while_wrapper.getControl().replaceAllUsesWith(new_while_wrapper.getControl());
  while_wrapper.erase();
  return new_while_op;
}

// Converts the control outputs of the while body to data outputs, thus
// removing control barrier at the end of while loop body.
void ConvertControlToDataOutputs(
    func::FuncOp while_body, SmallVectorImpl<TF::WhileOp>& while_callers,
    OperationSetTy& recompute_analysis_for_funcs,
    const TF::SideEffectAnalysis::Info& side_effect_analysis,
    const DataFlowSolver& solver, bool composite_tpuexecute_side_effects) {
  if (while_callers.empty()) return;

  // Collect access information for each resource in the while body that needs
  // to be chained, along with equivalence classes (resources in one class will
  // use the same chain).
  ResourceToOpsMapTy chain_resource_to_ops_map;
  llvm::EquivalenceClasses<ResourceAndDevice> resource_equivalence_classes;
  DeviceMap devices;
  CollectChainResources(
      while_body, chain_resource_to_ops_map, resource_equivalence_classes,
      devices, side_effect_analysis, solver, composite_tpuexecute_side_effects);

  // Check for presence of unknown side-effecting ops within the while loop
  // body. These ops act as barriers and the optimization would not yield much
  // inter iteration parallelism for this while loop body. So return with
  // warning.
  if (chain_resource_to_ops_map.count(kUnknownResourceAndDevice) > 0) {
    std::set<std::string> blocking_ops;
    for (Operation* op : chain_resource_to_ops_map[kUnknownResourceAndDevice]) {
      std::string op_name = op->getName().getStringRef().str();
      if (blocking_ops.insert(op_name).second) {
        LOG(WARNING)
            << "[`tf-executor-convert-control-to-data-outputs` disabled] "
               "Op type '"
            << op_name
            << "' has unknown side effects and blocks inter iteration "
               "parallelism for the while loop. Consider modeling side "
               "effects of this op.";
      }
    }
    return;
  }

  // First remove all control outputs of while loop body.
  SmallPtrSet<Operation*, 4> ops_connected_to_fetch;
  bool changed = RemoveAllControlOutputs(while_body, &ops_connected_to_fetch);

  // If there was no control output to be removed, return early.
  if (!changed) return;

  int num_chains = resource_equivalence_classes.getNumClasses();
  RankedTensorType chaining_data_type =
      RankedTensorType::get({}, OpBuilder(while_body).getI32Type());
  // Create new while body
  int num_old_outputs = while_body.getNumResults();
  AppendFunctionArguments(while_body, num_chains, chaining_data_type);
  AppendFunctionResults(while_body, num_chains, chaining_data_type);

  // Insert identity ops with control dep
  ChainResourceOps(while_body, chain_resource_to_ops_map,
                   resource_equivalence_classes, ops_connected_to_fetch,
                   num_old_outputs);
  // Modify all the while ops referencing the body function and the
  // corresponding while condition functions. Note that each while condition
  // needs to be modified only once.
  OperationSetTy visited;
  for (TF::WhileOp while_op : while_callers) {
    // If the while callers are modified as part of the optimization, then the
    // side effect analysis of their parent functions are invalidated. They
    // need to be recomputed.
    recompute_analysis_for_funcs.insert(
        while_op->getParentOfType<func::FuncOp>());
    func::FuncOp while_cond = while_op.cond_function();
    // Rewrite while op with extra chaining arguments and results.
    while_op = RewriteWhileOp(while_op, num_chains, chaining_data_type);
    bool first_visit = visited.insert(while_cond).second;
    if (!first_visit) continue;
    // Modify while condition function with extra chaining arguments.
    AppendFunctionArguments(while_cond, num_chains, chaining_data_type);
  }
}

void ConvertControlToDataOutputsPass::runOnOperation() {
  ModuleOp module = getOperation();

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  TF::LoadIsCompositeDataflowAnalysis(solver);
  if (failed(solver.initializeAndRun(module))) return signalPassFailure();

  // This pass assumes that all functions are suitable for export i.e., each
  // function has a single tf_executor.graph op and all islands wrap the
  // internal op perfectly. Verify that in the beginning once.
  if (failed(tensorflow::VerifyExportSuitable(module))) {
    signalPassFailure();
    return;
  }
  TF::SideEffectAnalysis side_effect_analysis(module);

  SymbolTableCollection table;
  SymbolUserMap symbol_map(table, module);
  llvm::SmallDenseMap<func::FuncOp, SmallVector<TF::WhileOp>>
      while_body_func_to_while_ops;

  // Get all the while body functions and the corresponding while ops first
  // because the symbol user map is invalidated once we start deleting while
  // ops.
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal()) continue;
    SmallVector<TF::WhileOp> while_callers = GetWhileCallers(func, symbol_map);
    if (while_callers.empty()) continue;
    while_body_func_to_while_ops[func] = while_callers;
    // TODO(b/295892728): verify while body sanity. This pass expects a 1:1
    // correspondence between function results and tf_executor.graph results.
  }
  // Keep track of functions whose side effect analysis is invalidated because
  // of modifications to that function.
  OperationSetTy recompute_analysis_for_funcs;

  for (auto& entry : while_body_func_to_while_ops) {
    func::FuncOp while_body = entry.getFirst();
    SmallVector<TF::WhileOp>& while_callers = entry.getSecond();
    if (recompute_analysis_for_funcs.contains(while_body)) {
      // TODO(b/202540801): Recomputing side effect analysis for the entire
      // module is wasteful. It would be better to just recompute analysis for
      // specific functions but the current side effect analysis interface
      // does not allow that.
      side_effect_analysis = TF::SideEffectAnalysis(module);
    }
    ConvertControlToDataOutputs(
        while_body, while_callers, recompute_analysis_for_funcs,
        side_effect_analysis.GetAnalysisForFunc(while_body), solver,
        composite_tpuexecute_side_effects_);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorConvertControlToDataOutputsPass() {
  return std::make_unique<ConvertControlToDataOutputsPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorConvertControlToDataOutputsPass(
    bool composite_tpuexecute_side_effects) {
  return std::make_unique<ConvertControlToDataOutputsPass>(
      composite_tpuexecute_side_effects);
}

}  // namespace tf_executor
}  // namespace mlir
