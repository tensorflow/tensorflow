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

#include "tensorflow/core/grappler/costs/utils.h"

#include <stddef.h>
#include <utility>

#include "third_party/eigen3/Eigen/Core"

#if GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#include "cuda/include/cudnn.h"
#endif

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

static OpInfo::TensorProperties UnknownInput() {
  OpInfo::TensorProperties input;
  input.set_dtype(DataType::DT_INVALID);
  input.mutable_shape()->set_unknown_rank(true);
  return input;
}

static std::vector<TensorProto> ExtractTensors(const AttrValue& attr_value) {
  std::vector<TensorProto> tensors;
  switch (attr_value.value_case()) {
    case AttrValue::kTensor: {
      tensors.push_back(attr_value.tensor());
      break;
    }
    case AttrValue::kList: {
      for (const auto& tensor_proto : attr_value.list().tensor()) {
        tensors.push_back(tensor_proto);
      }
      break;
    }
    default: {}
  }
  return tensors;
}

// Annotate the op_info inputs with extra information when possible (e.g. the
// input value if it's known statically).
static void ExtractExtraProperties(
    const NodeDef& node,
    const std::unordered_map<string, const NodeDef*>& name_to_node,
    OpInfo* op_info) {
  OpRegistry* op_registry = OpRegistry::Global();
  const OpDef* op_def = nullptr;
  auto s = op_registry->LookUpOpDef(node.op(), &op_def);
  if (!s.ok()) {
    op_def = nullptr;
  }

  for (int i = 0; i < node.input_size(); ++i) {
    const string input_name = node.input(i);
    CHECK(!input_name.empty());
    if (IsControlInput(input_name)) {
      continue;
    }
    TensorId input_tensor_id = ParseTensorName(input_name);
    const string input_node_name(input_tensor_id.first);

    auto iter = name_to_node.find(input_node_name);
    if (iter == name_to_node.end()) continue;
    const NodeDef* input_node = iter->second;

    if (i >= op_info->inputs_size()) {
      LOG(ERROR) << "OpInfo's inputs doesn't match the graph! OpInfo: "
                 << op_info->DebugString()
                 << "\nCurrent node: " << node.DebugString()
                 << "\nInput node: " << input_node->DebugString();
    }

    // The value attribute in Const input is useful for cost prediction.
    if (input_node->op() == "Const" && i < op_info->inputs_size()) {
      auto it = input_node->attr().find("value");
      if (it == input_node->attr().end()) continue;

      const AttrValue& attr_value = it->second;
      std::vector<TensorProto> tensors = ExtractTensors(attr_value);
      if (tensors.empty()) continue;

      const TensorProto& t = tensors[0];
      OpInfo::TensorProperties* input = op_info->mutable_inputs(i);
      *(input->mutable_value()) = t;

      // For filename input, the file size can also be useful.
      if (op_def && i < op_def->input_arg_size() &&
          op_def->input_arg(i).name().find("filename") != std::string::npos) {
        Tensor tensor;
        if (!tensor.FromProto(t)) {
          continue;
        }
        if (tensor.NumElements() != 1) {
          continue;
        }
        const string filename = tensor.scalar<string>()();

        Env* env = Env::Default();
        FileStatistics stat;
        Status s = env->Stat(filename, &stat);
        if (!s.ok()) {
          continue;
        }
        AttrValue attr;
        attr.set_i(stat.length);
        string attr_key = strings::StrCat("input_", i, "_filesize");
        (*op_info->mutable_attr())[attr_key] = attr;
      }
    }

    // When the input is a handle (e.g. look up table handle), the information
    // in the op itself is not sufficient to predict the op memory.
    if (op_def && i < op_def->input_arg_size() &&
        op_def->input_arg(i).name().find("handle") != std::string::npos) {
      string new_key = strings::StrCat("parent_", i, "_op");
      AttrValue attr;
      attr.set_s(input_node->op());
      (*op_info->mutable_attr())[new_key] = attr;
      // TODO(yuefengz): Only parent node's op name is copied. Copy inputs
      // and attributes when necessary.
    }
  }
}

std::vector<OpInfo::TensorProperties> FindInputFeatures(
    const NodeDef& node,
    const std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    const std::unordered_map<string, const NodeDef*>& name_to_node) {
  std::vector<OpInfo::TensorProperties> inputs;
  for (const auto& input_name : node.input()) {
    CHECK(!input_name.empty());
    TensorId input_tensor_id = ParseTensorName(input_name);
    const string input_node_name(input_tensor_id.first);
    const int output_index = input_tensor_id.second;

    // Skip control inputs.
    if (output_index == Graph::kControlSlot) {
      continue;
    }

    auto it = name_to_cost.find(input_node_name);
    if (it == name_to_cost.end() || output_index < 0) {
      inputs.push_back(UnknownInput());
    } else {
      const CostGraphDef::Node* input_cost = it->second;
      if (input_cost->output_info_size() == 0) {
        inputs.push_back(UnknownInput());
      } else {
        const CostGraphDef::Node::OutputInfo& output =
            input_cost->output_info(output_index);
        OpInfo::TensorProperties input;
        input.set_dtype(output.dtype());
        *input.mutable_shape() = output.shape();
        inputs.push_back(input);
      }
    }
  }

  return inputs;
}

DeviceProperties GetDeviceInfo(const string& device_str) {
  DeviceProperties unknown;
  unknown.set_type("UNKNOWN");

  DeviceNameUtils::ParsedName parsed;
  if (DeviceNameUtils::ParseFullName(device_str, &parsed)) {
    if (parsed.type == "GPU") {
      TfGpuId tf_gpu_id(parsed.id);
      CudaGpuId cuda_gpu_id;
      Status s = GpuIdManager::TfToCudaGpuId(tf_gpu_id, &cuda_gpu_id);
      if (!s.ok()) {
        // We are probably running simulation without linking cuda libraries.
        cuda_gpu_id = CudaGpuId(parsed.id);
      }
      return GetLocalGPUInfo(cuda_gpu_id);
    } else if (parsed.type == "CPU") {
      return GetLocalCPUInfo();
    }
  }
  return unknown;
}

DeviceProperties GetDeviceInfo(const CostGraphDef::Node& node) {
  return GetDeviceInfo(node.device());
}

OpInfo BuildOpInfoWithoutDevice(
    const NodeDef& node,
    const std::unordered_map<string, const NodeDef*>& name_to_node,
    const std::vector<OpInfo::TensorProperties>& inputs) {
  OpInfo op_info;
  op_info.set_op(node.op());
  *op_info.mutable_attr() = node.attr();
  for (auto& input : inputs) {
    *op_info.add_inputs() = input;
  }
  ExtractExtraProperties(node, name_to_node, &op_info);
  return op_info;
}

string GetOpDescription(const OpInfo& op_info) {
  string description = "[";
  description += "Op=" + op_info.op() + ", ";
  description += "input_shapes=[";
  for (auto const& input : op_info.inputs()) {
    description += PartialTensorShape::DebugString(input.shape());
  }
  description += "]";
  return description;
}

OpPerformanceList CostGraphToOpPerformanceData(const CostGraphDef& cost_graph,
                                               const GraphDef& graph) {
  OpPerformanceList ret;
  std::unordered_map<string, const CostGraphDef::Node*> name_to_cost;
  std::unordered_map<string, const NodeDef*> name_to_node;
  for (auto& node : cost_graph.node()) {
    name_to_cost[node.name()] = &node;
  }
  for (auto& node : graph.node()) {
    name_to_node[node.name()] = &node;
  }

  for (const auto& node : graph.node()) {
    // Skip the nodes that are not in the cost graph: these are nodes that
    // aren't run, because they aren't in the intersection of transitive
    // fan-in of a fetch node and the transitive fan-out of an input, or nodes
    // that were optimized away by the optimizer. Since they don't contribute
    // to the execution time we simply discard them.
    auto it = name_to_cost.find(node.name());
    if (it == name_to_cost.end()) {
      continue;
    }
    const CostGraphDef::Node* cost_node = it->second;

    OpPerformance* perf = ret.add_op_performance();
    perf->set_node(node.name());

    std::vector<OpInfo::TensorProperties> inputs =
        FindInputFeatures(node, name_to_cost, name_to_node);
    *perf->mutable_op() = BuildOpInfoWithoutDevice(node, name_to_node, inputs);
    *perf->mutable_op()->mutable_device() = GetDeviceInfo(cost_node->device());

    perf->set_temporary_memory_size(cost_node->temporary_memory_size());
    // Note that CostGraphDef::Node::compute_cost is microseconds, while
    // OpPerformance.compute_cost is nanoseconds.
    perf->set_compute_cost(cost_node->compute_cost() * 1000);
    perf->set_compute_time(cost_node->compute_time() * 1000);
    perf->set_memory_time(cost_node->memory_time() * 1000);

    for (const auto& output_info : cost_node->output_info()) {
      perf->mutable_op_memory()->add_output_memory(output_info.size());
    }

    perf->mutable_op_memory()->set_temp_memory(
        cost_node->temporary_memory_size());
    perf->mutable_op_memory()->set_persistent_memory(
        cost_node->persistent_memory_size());
  }
  return ret;
}

void TensorSizeHistogram::Add(const uint64 value) {
  num_elem_++;
  sum_elem_ += value;
  min_ = std::min(min_, value);
  max_ = std::max(max_, value);
  buckets_[Index(value)]++;
}

void TensorSizeHistogram::Merge(const TensorSizeHistogram& src) {
  num_elem_ += src.num_elem_;
  sum_elem_ += src.sum_elem_;
  min_ = std::min(min_, src.min_);
  max_ = std::max(max_, src.max_);
  std::transform(buckets_.begin(), buckets_.end(), src.buckets_.begin(),
                 buckets_.begin(), std::plus<uint64>());
}

std::string TensorSizeHistogram::ToString() const {
  std::string r;
  char buf[200];
  snprintf(buf, sizeof(buf), "Count: %lld, Average: ", num_elem_);
  r.append(buf);
  r.append(strings::HumanReadableNumBytes(Average()));
  r.append(", Min: ");
  r.append(strings::HumanReadableNumBytes(min_));
  r.append(", Max: ");
  r.append(strings::HumanReadableNumBytes(max_));
  r.append("\n------------------------------------------------------\n");
  const double mult = num_elem_ > 0 ? 100.0 / num_elem_ : 0.0;
  uint64 cumul_sum = 0;

  const int size_string_width = 12;
  for (int i = 0; i < buckets_.size(); i++) {
    if (buckets_[i] == 0) continue;
    cumul_sum += buckets_[i];
    r.append("[ ");
    if (i == 0) {
      r.append(size_string_width - 2, ' ');
      r.append("0B");
    } else {
      uint64 left = 1ULL << (i - 1);
      const auto left_string = strings::HumanReadableNumBytes(left);
      r.append(size_string_width - left_string.size(), ' ');
      r.append(left_string);
    }
    r.append(", ");
    uint64 right = 1ULL << i;
    const auto right_string = strings::HumanReadableNumBytes(right);
    r.append(size_string_width - right_string.size(), ' ');
    r.append(right_string);
    snprintf(buf, sizeof(buf), ") %7lld %7.3f%% %7.3f%% ",
             buckets_[i],         // count
             mult * buckets_[i],  // percentage
             mult * cumul_sum);   // cum percentage
    r.append(buf);

    // Add hash marks based on percentage; 40 marks for 100%.
    auto marks = static_cast<int>(
        (static_cast<double>(40 * buckets_[i] + (num_elem_ >> 1)) / num_elem_));
    r.append(marks, '#');
    r.push_back('\n');
  }
  return r;
}

const int TensorSizeHistogram::Index(const uint64 value) const {
  // Log2Floor64 returns -1 for 0, 0 for 1, 1 for 2-3, 2 for 4-7, ...
  const auto index = Log2Floor64(value) + 1;
  return std::min(index, kMaxBuckets - 1);
}

string GetDeviceClassForNonChannelDevice(const string& device_name) {
  DeviceNameUtils::ParsedName parsed_name;
  bool parsed = DeviceNameUtils::ParseFullName(device_name, &parsed_name);
  if (!parsed) {
    string name = str_util::StringReplace(device_name, "/job_", "/job:", true);
    name = str_util::StringReplace(name, "/replica_", "/replica:", true);
    name = str_util::StringReplace(name, "/task_", "/task:", true);
    name = str_util::StringReplace(name, "/device_", "/device:", true);
    name = str_util::StringReplace(name, "GPU_", "GPU:", true);
    name = str_util::StringReplace(name, "CPU_", "CPU:", true);
    name = str_util::StringReplace(name, "gpu_", "gpu:", true);
    name = str_util::StringReplace(name, "cpu_", "cpu:", true);
    parsed = DeviceNameUtils::ParseFullName(name, &parsed_name);
  }
  if (parsed) {
    const string jobname = parsed_name.has_job ? parsed_name.job : "";
    return strings::StrCat("/", jobname, "/", parsed_name.type);
  } else {
    return "Unclassified";
  }
}

string GetDeviceClass(const string& device_name) {
  // TODO(dyoon): channel device name follows the convention we currently have
  // in VirtualScheduler. This should be revised with VirtualScheduler as well
  // as VirtualPlacer in the future.
  if (device_name.find("Channel") != string::npos) {
    const string from = "_from_";
    const string to = "_to_";
    const auto from_loc = device_name.find(from);
    const auto to_loc = device_name.find(to);
    const auto src_device_full = device_name.substr(
        from_loc + from.size(), to_loc - (from_loc + from.size()));
    const auto dst_device_full = device_name.substr(to_loc + to.size());
    return strings::StrCat(
        "Channel", ": ", GetDeviceClassForNonChannelDevice(src_device_full),
        " -> ", GetDeviceClassForNonChannelDevice(dst_device_full));
  } else {
    return GetDeviceClassForNonChannelDevice(device_name);
  }
}

string GetStatsStringFromRunMetadata(const RunMetadata& run_metadata,
                                     bool verbosity) {
  // TODO(dyoon): print out other stats as needed.
  std::ostringstream output;

  // Tensor size histogram:
  // if verbosity, it outputs per-device histogram,
  // otherwise, only per-class histogram.
  std::unordered_map<string, TensorSizeHistogram> device_to_hist_map;
  const auto& step_stats = run_metadata.step_stats();
  for (const auto& dev_stat : step_stats.dev_stats()) {
    const auto& device_name = dev_stat.device();
    auto& hist = device_to_hist_map[device_name];
    for (const auto& node_stat : dev_stat.node_stats()) {
      for (const auto& node_output : node_stat.output()) {
        // TODO(dyoon): Calculate tensor size from tensor_description's dtype
        // and shape, instead of using optional allocation_description.
        const auto size = node_output.tensor_description()
                              .allocation_description()
                              .allocated_bytes();
        hist.Add(size);
      }
    }
  }
  if (verbosity) {
    output << "\n";
    output << "Per device tensor size histogram.\n";
  }

  std::unordered_map<string, TensorSizeHistogram> device_class_to_hist_map;
  for (const auto& device_hist : device_to_hist_map) {
    const auto& device_name = device_hist.first;
    const auto& hist = device_hist.second;
    if (verbosity) {
      output << "Device: " << device_name << "\n" << hist.ToString() << "\n";
    }
    const auto device_class = GetDeviceClass(device_name);
    auto it = device_class_to_hist_map.find(device_class);
    if (it == device_class_to_hist_map.end()) {
      device_class_to_hist_map.emplace(device_class, TensorSizeHistogram(hist));
    } else {
      it->second.Merge(hist);
    }
  }
  output << "\n";
  output << "Aggregated per device / channel type tensor size histogram:\n";
  for (const auto& device_hist : device_class_to_hist_map) {
    const auto& device_name = device_hist.first;
    const auto& hist = device_hist.second;
    output << "Device: " << device_name << "\n" << hist.ToString() << "\n";
  }
  output << "\n";

  return output.str();
}

}  // end namespace grappler
}  // end namespace tensorflow
