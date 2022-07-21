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
#include "tensorflow/core/data/captured_function.h"

#include <utility>

#include "absl/time/clock.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#endif  // !IS_MOBILE_PLATFORM

namespace tensorflow {
namespace data {
namespace {

constexpr char kAllowSmallFunctionOptimizations[] =
    "allow_small_function_optimizations";

// Simplistic implementation of the `StepStatsCollectorInterface` that only
// cares about collecting the CPU time needed to execute a captured function.
class SimpleStepStatsCollector : public StepStatsCollectorInterface {
 public:
  void IncrementProcessingTime(int64_t delta) {
    mutex_lock l(mu_);
    processing_time_ += delta;
  }

  NodeExecStatsInterface* CreateNodeExecStats(const NodeDef* node) override {
    return new SimpleNodeExecStats(this);
  }

  string ReportAllocsOnResourceExhausted(const string& err) override {
    return "";
  }

  int64_t processing_time() {
    tf_shared_lock l(mu_);
    return processing_time_;
  }

 private:
  class SimpleNodeExecStats : public NodeExecStatsInterface {
   public:
    explicit SimpleNodeExecStats(SimpleStepStatsCollector* step_stats_collector)
        : step_stats_collector_(step_stats_collector) {}

    void Done(const string& device) override {
      step_stats_collector_->IncrementProcessingTime(end_time_ns_ -
                                                     start_time_ns_);
      delete this;
    }

    void RecordExecutorStarted() override {
      start_time_ns_ = absl::GetCurrentTimeNanos();
    }

    void RecordComputeStarted() override {}

    void RecordComputeEnded() override {}

    void RecordExecutorEnded() override {
      end_time_ns_ = absl::GetCurrentTimeNanos();
    }

    bool TrackAllocations() const override { return false; }

    void SetMemory(OpKernelContext* ctx) override {}

    void SetOutput(int slot, const Tensor* tensor) override {}

    void SetScheduled(int64_t nanos) override {}

   private:
    int64_t start_time_ns_ = 0;
    int64_t end_time_ns_ = 0;
    SimpleStepStatsCollector* step_stats_collector_;  // Not owned.
  };

  mutex mu_;
  int64_t processing_time_ TF_GUARDED_BY(mu_) = 0;
};

Status GetCapturedInput(const CapturedFunction* const func, int index,
                        const Tensor** out) {
  if (TF_PREDICT_FALSE(index >= func->captured_inputs().size())) {
    return errors::OutOfRange(
        "Out of range access to captured inputs for function ",
        func->func().name(), ". Index: ", index,
        ". Num captured inputs: ", func->captured_inputs().size());
  }
  *out = &func->captured_inputs()[index];
  return OkStatus();
}

Status RunShortCircuit(const ShortCircuitInfo& info,
                       const std::vector<Tensor>& args,
                       const CapturedFunction* const func,
                       std::vector<Tensor>* rets) {
  VLOG(3) << "Running function " << func->func().name() << " short circuit";
  const int num_args = args.size();
  rets->reserve(info.indices.size());
  for (size_t i = 0; i < info.indices.size(); ++i) {
    if (info.indices[i] < num_args) {
      rets->push_back(args[info.indices[i]]);
    } else {
      const Tensor* captured_input;
      TF_RETURN_IF_ERROR(
          GetCapturedInput(func, info.indices[i] - num_args, &captured_input));
      rets->push_back(*captured_input);
    }
  }
  return OkStatus();
}

Status RunShortCircuit(const ShortCircuitInfo& info, std::vector<Tensor>&& args,
                       const CapturedFunction* const func,
                       std::vector<Tensor>* rets) {
  VLOG(3) << "Running function " << func->func().name() << " short circuit";
  const int num_args = args.size();
  rets->reserve(info.indices.size());
  for (size_t i = 0; i < info.indices.size(); ++i) {
    if (info.indices[i] < num_args) {
      if (info.can_move[i]) {
        rets->push_back(std::move(args[info.indices[i]]));
      } else {
        rets->push_back(args[info.indices[i]]);
      }
    } else {
      const Tensor* captured_input;
      TF_RETURN_IF_ERROR(
          GetCapturedInput(func, info.indices[i] - num_args, &captured_input));
      rets->push_back(*captured_input);
    }
  }
  return OkStatus();
}

Status CreateShortCircuitInfo(OpKernelConstruction* ctx,
                              const NameAttrList& func,
                              ShortCircuitInfo* info) {
  auto& indices = info->indices;

  FunctionLibraryRuntime::Handle fn_handle;
  TF_RETURN_IF_ERROR(ctx->function_library()->Instantiate(
      func.name(), AttrSlice(&func.attr()), &fn_handle));
  auto cleanup = gtl::MakeCleanup([ctx, fn_handle]() {
    Status s = ctx->function_library()->ReleaseHandle(fn_handle);
    if (!s.ok()) {
      LOG(WARNING) << "Failed to release handle: " << s.error_message();
    }
  });

  // If the function contains any stateful operations, we conservatively execute
  // the entire function.
  if (ctx->function_library()->IsStateful(func.name())) {
    return OkStatus();
  }

  const FunctionBody* fn_body =
      ctx->function_library()->GetFunctionBody(fn_handle);
  indices.resize(fn_body->ret_nodes.size());

  for (size_t i = 0; i < fn_body->ret_nodes.size(); ++i) {
    Node* ret_node = fn_body->ret_nodes[i];
    Node* ret_input_node;
    TF_RETURN_IF_ERROR(ret_node->input_node(0, &ret_input_node));

    while (ret_input_node->def().op() == "Identity") {
      TF_RETURN_IF_ERROR(ret_input_node->input_node(0, &ret_input_node));
    }

    if (ret_input_node->def().op() == FunctionLibraryDefinition::kArgOp) {
      TF_RETURN_IF_ERROR(
          GetNodeAttr(ret_input_node->def(), "index", &(indices[i])));
    } else {
      indices.clear();
      break;
    }
  }

  // Compute the `can_move` vector.
  if (!indices.empty()) {
    auto& can_move = info->can_move;
    std::map<int, int> last_use;
    for (size_t i = 0; i < indices.size(); ++i) {
      last_use[indices[i]] = i;
    }
    can_move.resize(indices.size());
    for (int i = 0, end = indices.size(); i < end; ++i) {
      can_move[i] = last_use[indices[i]] == i;
    }
  }

  return OkStatus();
}

Status CreateFunctionLibraryDefinition(
    const FunctionLibraryDefinition* lib_def, const string& func_name,
    std::unique_ptr<FunctionLibraryDefinition>* result) {
  DCHECK(lib_def != nullptr);
  const FunctionDef* fdef = lib_def->Find(func_name);
  if (TF_PREDICT_FALSE(fdef == nullptr)) {
    return errors::FailedPrecondition(strings::StrCat(
        "Could not find required function definition ", func_name));
  }
  *result = std::make_unique<FunctionLibraryDefinition>(
      lib_def->ReachableDefinitions(*fdef));
  return (*result)->CopyFunctionDefFrom(func_name, *lib_def);
}

Status LookupFunction(const FunctionLibraryDefinition& lib_def,
                      const string& name, const FunctionDef** fdef) {
  *fdef = lib_def.Find(name);
  if (*fdef == nullptr) {
    return errors::InvalidArgument(
        "Failed to find function ", name,
        " in function library: ", lib_def.ToProto().DebugString());
  }
  return OkStatus();
}

class CallFrameBase : public CallFrameInterface {
 public:
  explicit CallFrameBase(DataTypeSlice ret_types)
      : ret_types_(ret_types), retvals_(ret_types.size()) {}

  // Caller methods.
  Status ConsumeRetvals(std::vector<Tensor>* retvals) {
    retvals->reserve(retvals_.size());
    int i = 0;
    for (auto&& val : retvals_) {
      if (!val) {
        return errors::Internal("No return value for index ", i, ".");
      }
      retvals->emplace_back(std::move(val.value()));
      ++i;
    }
    return OkStatus();
  }

  size_t num_retvals() const override { return retvals_.size(); }

  // Callee methods.
  Status SetRetval(int index, const Tensor& val) override {
    const int retvals_size = retvals_.size();
    if (index < retvals_size && val.dtype() == ret_types_[index] &&
        !retvals_[index]) {
      retvals_[index] = val;
      return OkStatus();
    } else if (index >= retvals_size) {
      return errors::InvalidArgument("Return value ", index,
                                     " is out of range.");
    } else if (val.dtype() != ret_types_[index]) {
      return errors::InvalidArgument("Expected type ",
                                     DataTypeString(ret_types_[index]),
                                     " for return value ", index, " but got ",
                                     DataTypeString(val.dtype()), ".");
    } else {
      return errors::Internal("Attempted to set return value ", index,
                              " more than once.");
    }
  }

 private:
  DataTypeSlice ret_types_;
  std::vector<gtl::optional<Tensor>> retvals_;
  TF_DISALLOW_COPY_AND_ASSIGN(CallFrameBase);
};

class OwnedArgsCallFrame : public CallFrameBase {
 public:
  OwnedArgsCallFrame(std::vector<Tensor>&& args,
                     const std::vector<Tensor>* captured_inputs,
                     DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(std::move(args)),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, const Tensor** val) override {
    const int args_size = args_.size();
    const int captured_inputs_size = captured_inputs_->size();
    if (index < args_size) {
      *val = &args_[index];
      return OkStatus();
    } else if (index < args_size + captured_inputs_size) {
      *val = &(*captured_inputs_)[index - args_.size()];
      return OkStatus();
    } else {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    }
  }

  // Since we own the argument tensors in `args_`, we can implement
  // `ConsumeArg()` for those arguments.
  void ConsumeArg(int index, Tensor* val) override {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, args_.size());
    *val = std::move(args_[index]);
  }
  bool CanConsumeArg(int index) const override {
    return index >= 0 && index < static_cast<int>(args_.size());
  }

 private:
  std::vector<Tensor> args_;
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

class BorrowedArgsCallFrame : public CallFrameBase {
 public:
  BorrowedArgsCallFrame(const std::vector<Tensor>& args,
                        const std::vector<Tensor>* captured_inputs,
                        DataTypeSlice ret_types)
      : CallFrameBase(ret_types),
        args_(args),
        captured_inputs_(captured_inputs) {}

  size_t num_args() const override {
    return args_.size() + captured_inputs_->size();
  }

  // Callee methods.
  Status GetArg(int index, const Tensor** val) override {
    const int args_size = args_.size();
    const int captured_inputs_size = captured_inputs_->size();
    if (index < args_size) {
      *val = &args_[index];
      return OkStatus();
    } else if (index < args_size + captured_inputs_size) {
      *val = &(*captured_inputs_)[index - args_size];
      return OkStatus();
    } else {
      return errors::InvalidArgument("Argument ", index, " is out of range.");
    }
  }

 private:
  const std::vector<Tensor>& args_;                   // Not owned.
  const std::vector<Tensor>* const captured_inputs_;  // Not owned.
};

}  // namespace

Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const IteratorBase* parent,
    const std::vector<Tensor>& input_element, int64_t thread_index,
    const InstantiatedCapturedFunction& inst_captured_func, StringPiece prefix,
    std::unique_ptr<IteratorBase>* out_iterator) {
  return MakeIteratorFromInputElement(ctx, parent, input_element, thread_index,
                                      inst_captured_func, prefix, out_iterator,
                                      /*node=*/nullptr);
}

Status MakeIteratorFromInputElement(
    IteratorContext* ctx, const IteratorBase* parent,
    const std::vector<Tensor>& input_element, int64_t thread_index,
    const InstantiatedCapturedFunction& inst_captured_func, StringPiece prefix,
    std::unique_ptr<IteratorBase>* out_iterator,
    const std::shared_ptr<model::Node>& node) {
  std::vector<Tensor> return_values;

  TF_RETURN_IF_ERROR(inst_captured_func.RunWithBorrowedArgs(
      ctx, input_element, &return_values, node));

  if (!(return_values.size() == 1 && return_values[0].dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(return_values[0].shape()))) {
    return errors::InvalidArgument(
        "Function must return a single scalar of dtype DT_VARIANT.");
  }

  // Retrieve the dataset that was created in `f`.
  DatasetBase* returned_dataset;
  TF_RETURN_IF_ERROR(
      GetDatasetFromVariantTensor(return_values[0], &returned_dataset));

  // Create an iterator for the dataset that was returned by `f`.
  std::string iterator_prefix = strings::StrCat(prefix, "[", thread_index, "]");

  return returned_dataset->MakeIterator(MakeNestedIteratorContext(ctx), parent,
                                        iterator_prefix, out_iterator);
}

IteratorContext MakeNestedIteratorContext(IteratorContext* ctx) {
  // Strip out any split providers so that they don't apply to sub-iterators.
  if (ctx->split_providers().empty()) {
    return *ctx;
  }
  IteratorContext::Params params(ctx);
  params.split_providers.clear();
  return IteratorContext(std::move(params));
}

/* static */
Status FunctionMetadata::Create(
    OpKernelConstruction* ctx, const string& func_name, Params params,
    std::shared_ptr<FunctionMetadata>* out_metadata) {
  NameAttrList func;
  TF_RETURN_IF_ERROR(ctx->GetAttr(func_name, &func));
  return Create(ctx, std::move(func), params, out_metadata);
}

Status FunctionMetadata::Create(
    OpKernelConstruction* ctx, NameAttrList&& func, Params params,
    std::shared_ptr<FunctionMetadata>* out_metadata) {
  out_metadata->reset(new FunctionMetadata(std::move(func), params));
  TF_RETURN_IF_ERROR(CreateFunctionLibraryDefinition(
      ctx->function_library()->GetFunctionLibraryDefinition(),
      (*out_metadata)->func_.name(), &(*out_metadata)->lib_def_));
  TF_RETURN_IF_ERROR(CreateShortCircuitInfo(
      ctx, (*out_metadata)->func_, &(*out_metadata)->short_circuit_info_));
  const FunctionDef* fdef;
  TF_RETURN_IF_ERROR(LookupFunction(*(*out_metadata)->lib_def(),
                                    (*out_metadata)->func().name(), &fdef));

  auto attr = fdef->attr().find(FunctionLibraryDefinition::kIntsOnDeviceAttr);
  if (attr != fdef->attr().end() && attr->second.b()) {
    VLOG(1) << "Disabling multi-device execution for a function that uses the "
            << FunctionLibraryDefinition::kIntsOnDeviceAttr << " attribute.";
    (*out_metadata)->use_multi_device_function_ = false;
    return OkStatus();
  }
  auto validate_arg = [](const OpDef::ArgDef& arg) {
    if (!arg.number_attr().empty() || !arg.type_list_attr().empty()) {
      VLOG(1) << "Disabling multi-device execution for a function with "
              << "a vector argument " << arg.name() << ".";
      return false;
    }
    return true;
  };
  for (const auto& arg : fdef->signature().input_arg()) {
    if (!validate_arg(arg)) {
      (*out_metadata)->use_multi_device_function_ = false;
      return OkStatus();
    }
  }
  for (const auto& arg : fdef->signature().output_arg()) {
    if (!validate_arg(arg)) {
      (*out_metadata)->use_multi_device_function_ = false;
      return OkStatus();
    }
  }
  return OkStatus();
}

/* static */
Status CapturedFunction::Create(
    OpKernelContext* ctx, std::shared_ptr<const FunctionMetadata> metadata,
    const string& argument_name,
    std::unique_ptr<CapturedFunction>* out_function) {
  OpInputList inputs;
  TF_RETURN_IF_ERROR(ctx->input_list(argument_name, &inputs));
  std::vector<Tensor> captured_inputs(inputs.begin(), inputs.end());
  return Create(ctx, std::move(metadata), std::move(captured_inputs),
                out_function);
}

/* static */
Status CapturedFunction::Create(
    OpKernelContext* ctx, std::shared_ptr<const FunctionMetadata> metadata,
    std::vector<Tensor>&& captured_inputs,
    std::unique_ptr<CapturedFunction>* out_function) {
  *out_function = absl::WrapUnique(
      new CapturedFunction(std::move(metadata), std::move(captured_inputs)));
  return OkStatus();
}

Status CapturedFunction::AddToGraph(
    SerializationContext* ctx, DatasetBase::DatasetGraphDefBuilder* b,
    std::vector<Node*>* other_arguments,
    DataTypeVector* other_arguments_types) const {
  other_arguments->reserve(captured_inputs_.size());
  other_arguments_types->reserve(captured_inputs_.size());
  for (const Tensor& t : captured_inputs_) {
    Node* node;
    if (!ctx->is_graph_rewrite()) {
      TF_RETURN_IF_ERROR(b->AddDatasetOrTensor(ctx, t, &node));
    } else {
      TF_RETURN_IF_ERROR(b->AddPlaceholder(t, &node));
      DCHECK_NE(ctx->input_list(), nullptr);
      ctx->input_list()->emplace_back(node->name(), t);
    }
    other_arguments->emplace_back(node);
    other_arguments_types->emplace_back(t.dtype());
  }
  TF_RETURN_IF_ERROR(
      b->AddFunction(ctx, metadata_->func().name(), *metadata_->lib_def()));
  return OkStatus();
}

Status CapturedFunction::Instantiate(
    IteratorContext* ctx, std::unique_ptr<InstantiatedCapturedFunction>*
                              instantiated_captured_function) {
  return CapturedFunction::Instantiate(InstantiateCapturedFunctionParams(ctx),
                                       instantiated_captured_function);
}

// TODO(b/190831948): Check whether the function creates a resource and if so,
// produce a warning.
Status CapturedFunction::Instantiate(
    InstantiateCapturedFunctionParams params,
    std::unique_ptr<InstantiatedCapturedFunction>*
        instantiated_captured_function) {
  // The context's runtime will be used for all subsequent calls.
  FunctionLibraryRuntime* lib = params.flr;
  FunctionLibraryRuntime::InstantiateOptions inst_opts;
  inst_opts.lib_def = metadata_->lib_def();
  inst_opts.create_kernels_eagerly = true;
  inst_opts.default_device_to_target = metadata_->use_default_device();
  inst_opts.config_proto =
      lib->config_proto() ? *lib->config_proto() : ConfigProto();
  if (GetExperiments().contains(kAllowSmallFunctionOptimizations)) {
    inst_opts.allow_small_function_optimizations = true;
  } else {
    if (!metadata_->use_inter_op_parallelism()) {
      inst_opts.executor_type = "SINGLE_THREADED_EXECUTOR";
    }
  }
  inst_opts.is_multi_device_function = metadata_->use_multi_device_function();
  if (!params.function_handle_cache) {
    // If the caller does not provide a cache, we use the FLR cache.
    inst_opts.use_function_cache = true;
  }

  // We infer the target device from the function library runtime.
  DCHECK(lib->device() != nullptr);
  inst_opts.target = lib->device()->name();

  // Maps from a CompositeDevice name to underlying physical device names.
  absl::flat_hash_map<string, std::vector<string>> composite_devices;

  if (inst_opts.is_multi_device_function) {
    // Compute devices of non-captured inputs.
    //
    // We infer the number of non-captured inputs by subtracting the number
    // of captured inputs from the number of input arguments and we infer the
    // input devices from the function library runtime.
    const FunctionDef* fdef;
    TF_RETURN_IF_ERROR(
        LookupFunction(*metadata_->lib_def(), metadata_->func().name(), &fdef));
    size_t num_non_captured_inputs =
        fdef->signature().input_arg_size() - captured_inputs_.size();
    for (size_t i = 0; i < num_non_captured_inputs; ++i) {
      inst_opts.input_devices.push_back(inst_opts.target);
    }
    // Compute devices of captured inputs.
    // TODO(jsimsa): Correctly handle tensors on devices other than CPU:0.
    Device* cpu_device;
    TF_RETURN_IF_ERROR(lib->device_mgr()->LookupDevice("CPU:0", &cpu_device));
    std::unordered_map<int, DtypeAndPartialTensorShape>&
        input_resource_variable_dtypes_and_shapes =
            inst_opts.input_resource_dtypes_and_shapes;
    for (size_t i = 0; i < captured_inputs_.size(); ++i) {
      const auto& input = captured_inputs_[i];
      DataType dtype = input.dtype();
      if (dtype == DT_RESOURCE) {
        const auto& handles = input.flat<ResourceHandle>();
        const ResourceHandle& handle0 = handles(0);
        string composite_device;
        auto iter = fdef->arg_attr().find(num_non_captured_inputs + i);
        if (iter != fdef->arg_attr().end()) {
          auto arg_attr = iter->second.attr().find("_composite_device");
          if (arg_attr != iter->second.attr().end()) {
            composite_device = arg_attr->second.s();
          }
        }
        if (!composite_device.empty()) {
          if (composite_devices.find(composite_device) ==
              composite_devices.end()) {
            for (int i = 0; i < handles.size(); ++i) {
              composite_devices[composite_device].push_back(
                  handles(i).device());
            }
          }
          inst_opts.input_devices.push_back(composite_device);
        } else {
          inst_opts.input_devices.push_back(handle0.device());
        }
        const auto& dtypes_and_shapes = handle0.dtypes_and_shapes();
        // Set dtypes and shapes for resource variable inputs.
        if (!dtypes_and_shapes.empty()) {
          input_resource_variable_dtypes_and_shapes[num_non_captured_inputs +
                                                    i] =
              dtypes_and_shapes.at(0);
        }
      } else if (MTypeFromDType(dtype) == HOST_MEMORY) {
        inst_opts.input_devices.push_back(cpu_device->name());
      } else {
        // Fall back to using the function library runtime device.
        inst_opts.input_devices.push_back(inst_opts.target);
      }
    }

    for (const auto& it : composite_devices) {
      inst_opts.composite_devices[it.first] = &it.second;
    }

    for (int i = 0, end = fdef->signature().output_arg_size(); i < end; ++i) {
      inst_opts.output_devices.push_back(inst_opts.target);
    }

#if !defined(IS_MOBILE_PLATFORM)
    grappler::GrapplerItem::OptimizationOptions optimization_options;
    optimization_options.allow_pruning_stateful_and_dataset_ops = false;
    ConfigProto config_proto = inst_opts.config_proto;
    // Layout optimizations are excluded because they assume that ops without
    // explicit device assignment will be placed on GPU (if available) but
    // that's not the case for operations within tf.data functions.
    config_proto.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_layout_optimizer(RewriterConfig::OFF);
    // TODO(b/120437209): Re-enable constant folding.
    config_proto.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_constant_folding(RewriterConfig::OFF);
    inst_opts.optimize_graph_fn =
        std::bind(tensorflow::grappler::OptimizeGraph, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3,
                  std::placeholders::_4, std::placeholders::_5,
                  std::move(config_proto), fdef->signature().name(),
                  std::move(optimization_options), std::placeholders::_6);
#endif  // !IS_MOBILE_PLATFORM
  }

  FunctionLibraryRuntime::Handle f_handle;
  if (params.function_handle_cache) {
    TF_RETURN_IF_ERROR(params.function_handle_cache->Instantiate(
        metadata_->func().name(), AttrSlice(&metadata_->func().attr()),
        inst_opts, &f_handle));
  } else {
    TF_RETURN_IF_ERROR(lib->Instantiate(metadata_->func().name(),
                                        AttrSlice(&metadata_->func().attr()),
                                        inst_opts, &f_handle));
  }

  DataTypeVector ret_types;
  TF_RETURN_IF_ERROR(lib->GetRetTypes(f_handle, &ret_types));

  bool is_multi_device;
  TF_RETURN_IF_ERROR(IsMultiDevice(lib, &is_multi_device));
  *instantiated_captured_function = absl::WrapUnique(
      new InstantiatedCapturedFunction(lib, f_handle, std::move(ret_types),
                                       *params.runner, this, is_multi_device));
  return OkStatus();
}

Status CapturedFunction::CheckExternalState() const {
  for (const auto& name : lib_def()->ListFunctionNames()) {
    TF_RETURN_IF_ERROR(
        IsFunctionStateful(*lib_def(), *(lib_def()->Find(name))));
  }
  return OkStatus();
}

CapturedFunction::CapturedFunction(
    std::shared_ptr<const FunctionMetadata> metadata,
    std::vector<Tensor> captured_inputs)
    : metadata_(std::move(metadata)),
      captured_inputs_(std::move(captured_inputs)) {}

Status CapturedFunction::IsMultiDevice(FunctionLibraryRuntime* flr,
                                       bool* is_multi_device) const {
  if (!metadata_->use_multi_device_function()) {
    *is_multi_device = false;
    return OkStatus();
  }

  const FunctionDef* fdef;
  TF_RETURN_IF_ERROR(
      LookupFunction(*metadata_->lib_def(), metadata_->func().name(), &fdef));

  Device* current_device = flr->device();
  DeviceType current_device_type(current_device->device_type());
  DeviceNameUtils::ParsedName current_device_name;
  if (!DeviceNameUtils::ParseFullName(current_device->name(),
                                      &current_device_name)) {
    return errors::InvalidArgument("Failed to parse device name: ",
                                   current_device->name());
  }

  // Check if any of the captured inputs are placed on a device not compatible
  // with the current device. For non-captured inputs, we assume they are placed
  // on the current device.
  for (const auto& input : captured_inputs_) {
    DataType dtype = input.dtype();
    if (dtype == DT_RESOURCE) {
      const ResourceHandle& handle = input.flat<ResourceHandle>()(0);
      DeviceNameUtils::ParsedName resource_device_name;
      if (!DeviceNameUtils::ParseFullName(handle.device(),
                                          &resource_device_name)) {
        return errors::InvalidArgument("Failed to parse device name: ",
                                       handle.device());
      }
      if (!DeviceNameUtils::AreCompatibleDevNames(current_device_name,
                                                  resource_device_name)) {
        *is_multi_device = true;
        return OkStatus();
      }
    }
  }

  // Check if all ops could be placed on the current device.
  for (const auto& name : metadata_->lib_def()->ListFunctionNames()) {
    const FunctionDef* fdef;
    TF_RETURN_IF_ERROR(LookupFunction(*metadata_->lib_def(), name, &fdef));
    for (const auto& node : fdef->node_def()) {
      // Check if the op has a kernel available for the current device.
      if (!KernelDefAvailable(current_device_type, node)) {
        *is_multi_device = true;
        return OkStatus();
      }
      // If the op has a requested device, check if the requested device is
      // compatible with the current device.
      if (!node.device().empty()) {
        DeviceNameUtils::ParsedName node_device_name;
        if (!DeviceNameUtils::ParseFullName(node.device(), &node_device_name)) {
          return errors::InvalidArgument("Failed to parse device name: ",
                                         node.device());
        }
        if (!DeviceNameUtils::AreCompatibleDevNames(current_device_name,
                                                    node_device_name)) {
          *is_multi_device = true;
          return OkStatus();
        }
      }
    }
  }

  *is_multi_device = false;
  return OkStatus();
}

InstantiatedCapturedFunction::InstantiatedCapturedFunction(
    FunctionLibraryRuntime* lib, FunctionLibraryRuntime::Handle f_handle,
    DataTypeVector ret_types, std::function<void(std::function<void()>)> runner,
    CapturedFunction* captured_func, bool is_multi_device)
    : lib_(lib),
      f_handle_(f_handle),
      ret_types_(std::move(ret_types)),
      captured_runner_(std::move(runner)),
      captured_func_(captured_func),
      is_multi_device_(is_multi_device) {}

Status InstantiatedCapturedFunction::Run(IteratorContext* ctx,
                                         std::vector<Tensor>&& args,
                                         std::vector<Tensor>* rets) const {
  return Run(ctx, std::move(args), rets, /*node=*/nullptr);
}

Status InstantiatedCapturedFunction::Run(
    IteratorContext* ctx, std::vector<Tensor>&& args, std::vector<Tensor>* rets,
    const std::shared_ptr<model::Node>& node) const {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    return RunShortCircuit(info, std::move(args), captured_func_, rets);
  }

  FunctionLibraryRuntime::Options f_opts;
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  CancellationManager cancellation_manager(ctx->cancellation_manager());
  f_opts.cancellation_manager = &cancellation_manager;
  f_opts.collective_executor = ctx->collective_executor();

  std::shared_ptr<SimpleStepStatsCollector> stats_collector;
  if (node || ctx->stats_aggregator()) {
    stats_collector = std::make_shared<SimpleStepStatsCollector>();
  }
  const bool collect_usage = node && ctx->model();
  f_opts.stats_collector = stats_collector.get();

  OwnedArgsCallFrame frame(std::move(args), &captured_func_->captured_inputs(),
                           ret_types_);
  profiler::TraceMe activity(
      [&] {
        return profiler::TraceMeEncode("InstantiatedCapturedFunction::Run",
                                       {{"id", f_opts.step_id}});
      },
      profiler::TraceMeLevel::kInfo);
  if (node) {
    // Resource usage for function execution is gathered from the executor.
    // TODO(jsimsa): Factor out common code for Run, RunAsync, and
    // RunWithBorrowedArguments
    if (collect_usage) node->record_stop(EnvTime::NowNanos());
    TF_RETURN_IF_ERROR(lib_->RunSync(std::move(f_opts), f_handle_, &frame));
    if (ctx->stats_aggregator()) {
      string prefix_with_func_name = strings::StrCat(
          node->name(), stats_utils::kDelimiter, captured_func_->func().name());
      ctx->stats_aggregator()->AddToHistogram(
          stats_utils::ExecutionTimeHistogramName(prefix_with_func_name),
          {static_cast<float>(stats_collector->processing_time())},
          node->num_elements());
    }
    node->add_processing_time(stats_collector->processing_time());
    if (collect_usage) node->record_start(EnvTime::NowNanos());
  } else {
    TF_RETURN_IF_ERROR(lib_->RunSync(std::move(f_opts), f_handle_, &frame));
  }
  return frame.ConsumeRetvals(rets);
}

Status InstantiatedCapturedFunction::RunWithBorrowedArgs(
    IteratorContext* ctx, const std::vector<Tensor>& args,
    std::vector<Tensor>* ret) const {
  return RunWithBorrowedArgs(ctx, args, ret, /*node=*/nullptr);
}

Status InstantiatedCapturedFunction::RunWithBorrowedArgs(
    IteratorContext* ctx, const std::vector<Tensor>& args,
    std::vector<Tensor>* rets, const std::shared_ptr<model::Node>& node) const {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    return RunShortCircuit(info, args, captured_func_, rets);
  }

  FunctionLibraryRuntime::Options f_opts;
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = ctx->runner();
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  CancellationManager cancellation_manager(ctx->cancellation_manager());
  f_opts.cancellation_manager = &cancellation_manager;
  f_opts.collective_executor = ctx->collective_executor();

  std::shared_ptr<SimpleStepStatsCollector> stats_collector;
  if (node || ctx->stats_aggregator()) {
    stats_collector = std::make_shared<SimpleStepStatsCollector>();
  }
  const bool collect_usage = node && ctx->model();
  f_opts.stats_collector = stats_collector.get();

  BorrowedArgsCallFrame frame(args, &captured_func_->captured_inputs(),
                              ret_types_);
  profiler::TraceMe activity(
      [&] {
        return profiler::TraceMeEncode(
            "InstantiatedCapturedFunction::RunWithBorrowedArgs",
            {{"id", f_opts.step_id}});
      },
      profiler::TraceMeLevel::kInfo);
  if (node) {
    // Resource usage for function execution is gathered from the executor.
    if (collect_usage) node->record_stop(EnvTime::NowNanos());
    TF_RETURN_IF_ERROR(lib_->RunSync(std::move(f_opts), f_handle_, &frame));
    if (ctx->stats_aggregator()) {
      string prefix_with_func_name = strings::StrCat(
          node->name(), stats_utils::kDelimiter, captured_func_->func().name());
      ctx->stats_aggregator()->AddToHistogram(
          stats_utils::ExecutionTimeHistogramName(prefix_with_func_name),
          {static_cast<float>(stats_collector->processing_time())},
          node->num_elements());
    }
    node->add_processing_time(stats_collector->processing_time());
    if (collect_usage) node->record_start(EnvTime::NowNanos());
  } else {
    TF_RETURN_IF_ERROR(lib_->RunSync(std::move(f_opts), f_handle_, &frame));
  }
  return frame.ConsumeRetvals(rets);
}

Status InstantiatedCapturedFunction::RunInstantiated(
    const std::vector<Tensor>& args, std::vector<Tensor>* rets) {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    return RunShortCircuit(info, args, captured_func_, rets);
  }

  FunctionLibraryRuntime::Options f_opts;
  ScopedStepContainer step_container(
      f_opts.step_id, [this](const string& name) {
        lib_->device()->resource_manager()->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = &step_container;
  f_opts.runner = &captured_runner_;
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  CancellationManager cancellation_manager;
  f_opts.cancellation_manager = &cancellation_manager;

  BorrowedArgsCallFrame frame(args, &captured_func_->captured_inputs(),
                              ret_types_);
  profiler::TraceMe activity(
      [&] {
        return profiler::TraceMeEncode(
            "InstantiatedCapturedFunction::RunInstantiated",
            {{"id", f_opts.step_id}});
      },
      profiler::TraceMeLevel::kInfo);
  TF_RETURN_IF_ERROR(lib_->RunSync(std::move(f_opts), f_handle_, &frame));
  return frame.ConsumeRetvals(rets);
}

void InstantiatedCapturedFunction::RunAsync(
    IteratorContext* ctx, std::vector<Tensor>&& args, std::vector<Tensor>* rets,
    FunctionLibraryRuntime::DoneCallback done,
    const std::shared_ptr<model::Node>& node) const {
  auto& info = captured_func_->short_circuit_info();
  if (!info.indices.empty()) {
    // Run the `done` callback on a threadpool thread, because it will
    // potentially do a non-trivial amount of (e.g. copying) work, and we may
    // want to run that concurrently with the next invocation.
    Status s = RunShortCircuit(info, std::move(args), captured_func_, rets);
    (*ctx->runner())(
        std::bind([s](FunctionLibraryRuntime::DoneCallback& done) { done(s); },
                  std::move(done)));
    return;
  }

  // NOTE(mrry): This method does not transfer ownership of `ctx`, and it may
  // be deleted before `done` is called. Take care not to capture `ctx` in any
  // code that may execute asynchronously in this function.
  OwnedArgsCallFrame* frame = new OwnedArgsCallFrame(
      std::move(args), &captured_func_->captured_inputs(), ret_types_);

  FunctionLibraryRuntime::Options f_opts;
  ResourceMgr* resource_mgr = lib_->device()->resource_manager();
  ScopedStepContainer* step_container = new ScopedStepContainer(
      f_opts.step_id, [resource_mgr](const string& name) {
        resource_mgr->Cleanup(name).IgnoreError();
      });
  f_opts.step_container = step_container;
  f_opts.runner = ctx->runner();
  f_opts.create_rendezvous = ShouldCreateRendezvous();
  auto cancellation_manager =
      std::make_unique<CancellationManager>(ctx->cancellation_manager());
  f_opts.cancellation_manager = cancellation_manager.get();
  f_opts.collective_executor = ctx->collective_executor();

  std::shared_ptr<SimpleStepStatsCollector> stats_collector;
  if (node || ctx->stats_aggregator()) {
    stats_collector = std::make_shared<SimpleStepStatsCollector>();
  }
  const bool collect_usage = node && ctx->model();
  f_opts.stats_collector = stats_collector.get();

  // Transfer ownership of the cancellation manager to `callback`.
  CancellationManager* raw_cancellation_manager =
      cancellation_manager.release();
  auto callback = std::bind(
      [this, rets, step_container, raw_cancellation_manager, frame, node,
       collect_usage](
          const FunctionLibraryRuntime::DoneCallback& done,
          IteratorContext* ctx,
          const std::shared_ptr<SimpleStepStatsCollector>& stats_collector,
          // Begin unbound arguments.
          Status s) {
        delete step_container;
        delete raw_cancellation_manager;
        if (s.ok()) {
          s = frame->ConsumeRetvals(rets);
        }
        delete frame;
        if (node) {
          // TODO(b/129085499) Utilize the `node_name` which would be unique
          // than the prefix for the function execution time statistics.
          // prefix_with_func_name would then be node_name + func_name.
          if (ctx->stats_aggregator()) {
            string prefix_with_func_name =
                strings::StrCat(node->name(), stats_utils::kDelimiter,
                                captured_func_->func().name());
            ctx->stats_aggregator()->AddToHistogram(
                stats_utils::ExecutionTimeHistogramName(prefix_with_func_name),
                {static_cast<float>(stats_collector->processing_time())},
                node->num_elements());
          }
          node->add_processing_time(stats_collector->processing_time());
        }
        if (collect_usage) {
          node->record_start(EnvTime::NowNanos());
        }
        done(s);
        if (collect_usage) {
          node->record_stop(EnvTime::NowNanos());
        }
      },
      std::move(done), ctx, std::move(stats_collector), std::placeholders::_1);

  profiler::TraceMe activity(
      [&] {
        return profiler::TraceMeEncode("InstantiatedCapturedFunction::RunAsync",
                                       {{"id", f_opts.step_id}});
      },
      profiler::TraceMeLevel::kInfo);
  // Stop the usage collection before calling `Run()` because `callback` may
  // be executed synchronously, and so the `node->record_start()` call within
  // `callback` would violate nesting.
  if (collect_usage) node->record_stop(EnvTime::NowNanos());
  lib_->Run(f_opts, f_handle_, frame, std::move(callback));
  if (collect_usage) node->record_start(EnvTime::NowNanos());
}

bool InstantiatedCapturedFunction::ShouldCreateRendezvous() const {
  // Rendezvous should only be created by the FLR for non-CPU single-device
  // functions. For multi-device functions the appropriate rendezvous will be
  // created by the process FLR.
  return lib_->device()->device_type() != DEVICE_CPU && !is_multi_device_;
}

}  // namespace data
}  // namespace tensorflow
