/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_node.h"

#include "tensorflow/core/profiler/internal/tfprof_utils.h"

namespace tensorflow {
namespace tfprof {
namespace {
bool CountAsAcceleratorTime(const string& device) {
  return device.find("stream:all") != device.npos;
}

bool CountAsCPUTime(const string& device) {
  return RE2::FullMatch(device, ".*/(gpu|cpu):\\d+");
}

bool IsCanonicalDevice(const string& device) { return CountAsCPUTime(device); }

}  // namespace
// Notes about start and end time from the NodeExecStats proto:
// For GPU, there is no difference between op_end_rel_micros and
// all_end_rel_micros. All are kernel times.
// For CPU, op_end_rel is the kernel time, while all_end_rel_micros includes
// some post-processing. Besides, currently, there is no way to measure
// the execution time of async ops accurately.
//
// Notes about device:
// For ops on gpu:
// It will appear in three different devices in RunMetadata: 1) gpu:x,
// 2) gpu:x:stream:all and 3) gpu:x:stream:id. 2) is used a combined view
// of all different 3). 1) is the op scheduling, pre-processing and
// post processing time. 3) is the execution time of GPU kernels on a stream.
// For ops on cpu:
// It will only appear as cpu:0.

void ExecStep::AddTimeStats(const string& dev, const NodeExecStats& step_stat) {
  devices_.insert(dev);
  if (step_stat.all_start_micros() > 0) {
    if (all_start_micros_ > 0) {
      all_start_micros_ = std::min(
          all_start_micros_, static_cast<int64>(step_stat.all_start_micros()));
    } else {
      all_start_micros_ = step_stat.all_start_micros();
    }
    int64 op_end_rel_micros = step_stat.op_end_rel_micros();
    // Round quick execution to 1 micro to be semantically robust.
    if (op_end_rel_micros == 0) {
      ++op_end_rel_micros;
    }
    latest_end_micros_ = std::max(
        latest_end_micros_, step_stat.all_start_micros() + op_end_rel_micros);

    const std::pair<int64, int64> pair =
        std::make_pair(step_stat.all_start_micros(), op_end_rel_micros);
    if (CountAsAcceleratorTime(dev)) {
      accelerator_execs_[dev].push_back(pair);
      op_execs_[dev].push_back(pair);
    } else if (CountAsCPUTime(dev)) {
      cpu_execs_[dev].push_back(pair);
      op_execs_[dev].push_back(pair);
      // In while-loop, a graph node is executed multiple times under
      // the same name.
      run_count_ += 1;
    }
  }
}

void ExecStep::AddMemoryStats(const string& dev,
                              const NodeExecStats& step_stat) {
  if (mem_initiated_) {
    return;
  }
  mem_initiated_ = true;

  for (const auto& mem : step_stat.memory()) {
    // TODO(xpan): Fix this hack. Currently the allocator name seems quite
    // ad-hoc.
    if (mem.allocator_name().find("GPU") == mem.allocator_name().npos) {
      continue;
    }
    allocator_bytes_in_use_ =
        std::max(allocator_bytes_in_use_,
                 static_cast<int64>(mem.allocator_bytes_in_use()));
  }
  int64 total_output_bytes = 0;
  for (const auto& output : step_stat.output()) {
    if (output.has_tensor_description() &&
        output.tensor_description().has_allocation_description()) {
      // TODO(xpan): Maybe allocated_bytes.
      int64 output_bytes = std::max(output.tensor_description()
                                        .allocation_description()
                                        .allocated_bytes(),
                                    output.tensor_description()
                                        .allocation_description()
                                        .requested_bytes());
      uint64 output_ptr =
          output.tensor_description().allocation_description().ptr();
      total_output_bytes += output_bytes;
      output_bytes_[output.slot()] = std::make_pair(output_bytes, output_ptr);
    }
  }
  if (step_stat.has_memory_stats()) {
    host_temp_bytes_ += step_stat.memory_stats().host_temp_memory_size();
    host_persistent_bytes_ +=
        step_stat.memory_stats().host_persistent_memory_size();
    accelerator_temp_bytes_ +=
        step_stat.memory_stats().device_temp_memory_size();
    accelerator_persistent_bytes_ +=
        step_stat.memory_stats().device_persistent_memory_size();
  }
  requested_bytes_ = total_output_bytes;
}

void TFGraphNode::AddStepStat(int64 step, const string& device,
                              const NodeExecStats& step_stat) {
  string dev = str_util::Lowercase(device);

  // TODO(xpan): Make this more robust?
  // See run_metadata_test.py
  // It can be /job:0/replica:0/xxxx/gpu:0, or simply /gpu:0.
  // It can has some ad-hoc suffix, such as /stream:xx or /memcpy:xx.
  if (IsCanonicalDevice(device)) {
    if (!canonical_device_.empty()) {
      if (canonical_device_ != dev) {
        fprintf(stderr, "Unexpected: graph node changed device: %s->%s.\n",
                canonical_device_.c_str(), dev.c_str());
        return;
      }
    } else {
      canonical_device_ = dev;
      // TODO(xpan): Support things other than gpu?
      host_device_ = StringReplace(dev, "gpu:\\d+", "cpu:0");
      AddOpType(canonical_device_);
    }
  }

  auto exec = execs_.find(step);
  if (exec == execs_.end()) {
    execs_.insert(std::pair<int64, ExecStep>(step, ExecStep(this)));
    exec = execs_.find(step);
  }

  exec->second.AddTimeStats(dev, step_stat);

  if (dev == canonical_device_) {
    exec->second.AddMemoryStats(dev, step_stat);
  }
}

int64 ExecStep::exec_micros() const {
  return accelerator_exec_micros() + cpu_exec_micros();
}

int64 ExecStep::accelerator_exec_micros() const {
  int64 total = 0;
  // Normally, an op should only be scheduled on 1 accelerator device.
  // Hence there should generally be 1 element in accelerator_execs_.
  for (const auto& execs : accelerator_execs_) {
    // An op can fire multiple kernels or
    // being scheduled multiple times in while-loop.
    for (const auto& exec : execs.second) {
      total += exec.second;
    }
  }
  return total;
}

int64 ExecStep::cpu_exec_micros() const {
  int64 total = 0;
  // Normally, an op can only be scheduled on 1 device.
  for (const auto& execs : cpu_execs_) {
    // An op can be scheduled multiple times in while-loop.
    for (const auto& exec : execs.second) {
      total += exec.second;
    }
  }
  return total;
}

std::vector<int64> ShapeProtoToVec(const TensorShapeProto& shape_pb) {
  std::vector<int64> shape_vec;
  if (shape_pb.dim_size() == 0 && !shape_pb.unknown_rank()) {
    // Scalar parameter with empty shape but known rank.
    shape_vec.push_back(1);
  } else {
    for (const auto& d : shape_pb.dim()) {
      shape_vec.push_back(d.size());
    }
  }
  return shape_vec;
}

TensorShapeProto VecToShapeProto(const std::vector<int64> shape_vec) {
  TensorShapeProto shape_pb;
  if (shape_vec.empty()) {
    shape_pb.set_unknown_rank(true);
    return shape_pb;
  }
  for (const int64 s : shape_vec) {
    shape_pb.add_dim()->set_size(s);
  }
  return shape_pb;
}

bool IsPlacedOnAccelerator(const string& device) {
  return device.find("gpu") != device.npos;
}
}  // namespace tfprof
}  // namespace tensorflow
