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
bool CountAsAcceleratorTime(const string& device) {
  return device.find("stream:all") != device.npos;
}
bool CountAsCPUTime(const string& device) {
  return RE2::FullMatch(device, ".*/(device:gpu|gpu|device:cpu|cpu):\\d+");
}
bool IsCanonicalDevice(const string& device) { return CountAsCPUTime(device); }

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
    if (exec_.all_start_micros() > 0) {
      exec_.set_all_start_micros(
          std::min(static_cast<int64>(exec_.all_start_micros()),
                   static_cast<int64>(step_stat.all_start_micros())));
    } else {
      exec_.set_all_start_micros(step_stat.all_start_micros());
    }
    int64 op_end_rel_micros = step_stat.op_end_rel_micros();
    // Round quick execution to 1 micro to be semantically robust.
    if (op_end_rel_micros == 0) {
      ++op_end_rel_micros;
    }
    exec_.set_latest_end_micros(
        std::max(static_cast<int64>(exec_.latest_end_micros()),
                 step_stat.all_start_micros() + op_end_rel_micros));

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
      exec_.set_run_count(exec_.run_count() + 1);
    }
  }
}

void ExecStep::AddMemoryStats(const string& dev,
                              const NodeExecStats& step_stat) {
  ExecMemory exec_mem;
  if (step_stat.all_start_micros() > 0) {
    exec_mem.set_memory_micros(step_stat.all_start_micros() +
                               step_stat.op_end_rel_micros());
  } else {
    absl::FPrintF(stderr, "%s has no start time, skipping\n",
                  step_stat.node_name());
    return;
  }

  int accelerator_allocator_cnt = 0;
  for (const auto& mem : step_stat.memory()) {
    // TODO(xpan): Fix this hack. Currently the allocator name seems quite
    // ad-hoc.
    if (mem.allocator_name().find("GPU") == mem.allocator_name().npos) {
      continue;
    }
    ++accelerator_allocator_cnt;
    exec_mem.set_allocator_bytes_in_use(
        std::max(static_cast<int64>(exec_mem.allocator_bytes_in_use()),
                 static_cast<int64>(mem.allocator_bytes_in_use())));
    for (const auto& alloc : mem.allocation_records()) {
      allocations_.push_back(alloc);
    }
  }
  if (accelerator_allocator_cnt > 1) {
    absl::FPrintF(stderr, "found %d gpu allocator for 1 node\n",
                  accelerator_allocator_cnt);
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

      auto& mem = (*exec_mem.mutable_output_memory())[output.slot()];
      mem.set_ptr(output_ptr);
      mem.set_bytes(output_bytes);
    }
  }
  exec_mem.set_output_bytes(total_output_bytes);

  if (step_stat.has_memory_stats()) {
    if (IsPlacedOnCPU(dev)) {
      // Currently we assume ops placed on gpu only allocate memory on gpu.
      exec_mem.set_host_temp_bytes(exec_mem.host_temp_bytes() +
                                   step_stat.memory_stats().temp_memory_size());
      exec_mem.set_host_persistent_bytes(
          exec_mem.host_persistent_bytes() +
          step_stat.memory_stats().persistent_memory_size());
    } else {
      exec_mem.set_accelerator_temp_bytes(
          exec_mem.accelerator_temp_bytes() +
          step_stat.memory_stats().temp_memory_size());
      exec_mem.set_accelerator_persistent_bytes(
          exec_mem.accelerator_persistent_bytes() +
          step_stat.memory_stats().persistent_memory_size());
    }
  }

  // TODO(xpan): Make this more accurate:
  // High level: Memory tracking is suspicious and requires large scale
  // clean up.
  // Investigate the memory usage difference between CPU/GPU with OpViewTest.
  //
  // 1. OpKernelConstruction::allocate_xxx is not traced. Below, we only
  //    discuss OpKernelContext-related allocations.
  // 2. allocate_output calls allocate_tensor, which is properly tracked in
  //    'NodeExecStats.memory'.
  // 3. allocate_temp is only tracked through record_xxx_temp. It appears
  //    in 'NodeExecStats.memory_stats'.
  // 4. allocate_persistent calls allocate_tensor, which is properly tracked
  //    in 'NodeExecStats.memory'. However, there is no way to count it as
  //    persistent now.
  // 5. record_xxx_persistent is called when allocate_persistent
  //    is not used and hence tracks some complementary bytes. It appears in
  //    'NodeExecStats.memory_stats'. It's suspicious. But we should
  //    use it now since it covers constant op.
  int64 residual_bytes = 0;
  int64 requested_bytes = 0;
  int64 peak_bytes = 0;
  for (const auto& mem : step_stat.memory()) {
    residual_bytes += mem.live_bytes();
    requested_bytes += mem.total_bytes();
    peak_bytes += mem.peak_bytes();
  }
  residual_bytes += exec_mem.host_persistent_bytes() +
                    exec_mem.accelerator_persistent_bytes();
  requested_bytes += exec_mem.host_persistent_bytes() +
                     exec_mem.accelerator_persistent_bytes() +
                     exec_mem.host_temp_bytes() +
                     exec_mem.accelerator_temp_bytes();
  peak_bytes += exec_mem.host_persistent_bytes() +
                exec_mem.accelerator_persistent_bytes() +
                exec_mem.host_temp_bytes() + exec_mem.accelerator_temp_bytes();

  exec_mem.set_requested_bytes(requested_bytes);
  exec_mem.set_residual_bytes(residual_bytes);
  exec_mem.set_peak_bytes(peak_bytes);
  memory_execs_.emplace_back(exec_mem);
}

void TFGraphNode::AddStepStat(int64 step, const string& device,
                              const NodeExecStats& step_stat) {
  string dev = absl::AsciiStrToLower(device);

  // TODO(xpan): Make this more robust?
  // See run_metadata_test.py
  // It can be /job:0/replica:0/xxxx/device:GPU:0, or simply /device:GPU:0.
  // It can has some ad-hoc suffix, such as /stream:xx or /memcpy:xx.
  if (IsCanonicalDevice(dev)) {
    if (!node_.canonical_device().empty()) {
      if (node_.canonical_device() != dev) {
        // TODO(xpan): Some RunMetadata node appears at multiple devices.
        // Need to address it.
        return;
      }
    } else {
      node_.set_canonical_device(dev);
      // TODO(xpan): Support things other than gpu?
      node_.set_host_device(StringReplace(dev, "gpu:\\d+", "cpu:0"));
      AddOpType(node_.canonical_device());
    }
  }

  auto exec = execs_.find(step);
  if (exec == execs_.end()) {
    execs_.insert(std::pair<int64, ExecStep>(step, ExecStep()));
    exec = execs_.find(step);
  }

  exec->second.AddTimeStats(dev, step_stat);

  if (dev == node_.canonical_device()) {
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

TensorShapeProto VecToShapeProto(const std::vector<int64>& shape_vec) {
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
bool IsPlacedOnCPU(const string& device) {
  return device.find("cpu") != device.npos;
}
}  // namespace tfprof
}  // namespace tensorflow
