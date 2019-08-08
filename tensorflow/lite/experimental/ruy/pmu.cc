/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/pmu.h"

#include "tensorflow/lite/experimental/ruy/check_macros.h"

#ifdef __linux__
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstdio>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ruy {

// Linux-specific. Not ARM-specific.
#ifdef __linux__
class PerfEvent {
 public:
  void Start(std::uint32_t type, std::uint64_t config) {
    perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.type = type;
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    fd_ = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
    if (fd_ == -1) {
      fprintf(stderr, "perf_event_open failed for config 0x%lx\n", config);
      // abort();
    }
    ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
  }

  void Stop() {
    ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
    RUY_CHECK_NE(read(fd_, &count_, sizeof(count_)), -1);
    close(fd_);
  }

  std::int64_t Count() const { return count_; }

 private:
  std::int64_t count_ = -1;
  int fd_ = -1;
};
#else
// Placeholder implementation to at least compile outside of linux.
#define PERF_TYPE_RAW 0
class PerfEvent {
 public:
  void Start(std::uint32_t, std::uint64_t) {}
  void Stop() {}
  std::int64_t Count() const { return 0; }
};
#endif

// ARM-specific. Query ARM PMU counters as Linux perf events using
// PERF_TYPE_RAW.
namespace arm_pmuv3 {
// These event numbers are listed in the ARMv8 architecture reference manual.
constexpr std::uint16_t L1I_CACHE_REFILL = 0x01;
constexpr std::uint16_t L1I_TLB_REFILL = 0x02;
constexpr std::uint16_t L1D_CACHE_REFILL = 0x03;
constexpr std::uint16_t L1D_CACHE = 0x04;
constexpr std::uint16_t L1D_TLB_REFILL = 0x05;
constexpr std::uint16_t LD_RETIRED = 0x06;
constexpr std::uint16_t ST_RETIRED = 0x07;
constexpr std::uint16_t INST_RETIRED = 0x08;
constexpr std::uint16_t EXC_TAKEN = 0x09;
constexpr std::uint16_t EXC_RETURN = 0x0A;
constexpr std::uint16_t CID_WRITE_RETIRED = 0x0B;
constexpr std::uint16_t PC_WRITE_RETIRED = 0x0C;
constexpr std::uint16_t BR_IMMED_RETIRED = 0x0D;
constexpr std::uint16_t BR_RETURN_RETIRED = 0x0E;
constexpr std::uint16_t UNALIGNED_LDST_RETIRED = 0x0F;
constexpr std::uint16_t BR_MIS_PRED = 0x10;
constexpr std::uint16_t CPU_CYCLES = 0x11;
constexpr std::uint16_t BR_PRED = 0x12;
constexpr std::uint16_t MEM_ACCESS = 0x13;
constexpr std::uint16_t L1I_CACHE = 0x14;
constexpr std::uint16_t L1D_CACHE_WB = 0x15;
constexpr std::uint16_t L2D_CACHE = 0x16;
constexpr std::uint16_t L2D_CACHE_REFILL = 0x17;
constexpr std::uint16_t L2D_CACHE_WB = 0x18;
constexpr std::uint16_t BUS_ACCESS = 0x19;
constexpr std::uint16_t MEMORY_ERROR = 0x1A;
constexpr std::uint16_t INST_SPEC = 0x1B;
constexpr std::uint16_t TTBR_WRITE_RETIRED = 0x1C;
constexpr std::uint16_t BUS_CYCLES = 0x1D;
constexpr std::uint16_t CHAIN = 0x1E;
constexpr std::uint16_t L1D_CACHE_ALLOCATE = 0x1F;
constexpr std::uint16_t L2D_CACHE_ALLOCATE = 0x20;
constexpr std::uint16_t BR_RETIRED = 0x21;
constexpr std::uint16_t BR_MIS_PRED_RETIRED = 0x22;
constexpr std::uint16_t STALL_FRONTEND = 0x23;
constexpr std::uint16_t STALL_BACKEND = 0x24;
constexpr std::uint16_t L1D_TLB = 0x25;
constexpr std::uint16_t L1I_TLB = 0x26;
constexpr std::uint16_t L2I_CACHE = 0x27;
constexpr std::uint16_t L2I_CACHE_REFILL = 0x28;
constexpr std::uint16_t L3D_CACHE_ALLOCATE = 0x29;
constexpr std::uint16_t L3D_CACHE_REFILL = 0x2A;
constexpr std::uint16_t L3D_CACHE = 0x2B;
constexpr std::uint16_t L3D_CACHE_WB = 0x2C;
constexpr std::uint16_t L2D_TLB_REFILL = 0x2D;
constexpr std::uint16_t L2I_TLB_REFILL = 0x2E;
constexpr std::uint16_t L2D_TLB = 0x2F;
constexpr std::uint16_t L2I_TLB = 0x30;
constexpr std::uint16_t LL_CACHE = 0x32;
constexpr std::uint16_t LL_CACHE_MISS = 0x33;
constexpr std::uint16_t DTLB_WALK = 0x34;
constexpr std::uint16_t LL_CACHE_RD = 0x36;
constexpr std::uint16_t LL_CACHE_MISS_RD = 0x37;

// Additional implementation-defined events found by googling around.
constexpr std::uint16_t L1D_CACHE_RD = 0x40;
constexpr std::uint16_t L1D_CACHE_REFILL_RD = 0x42;
constexpr std::uint16_t L1D_TLB_REFILL_RD = 0x4C;
constexpr std::uint16_t L1D_TLB_RD = 0x4E;
constexpr std::uint16_t L2D_CACHE_RD = 0x50;
constexpr std::uint16_t L2D_CACHE_REFILL_RD = 0x52;
constexpr std::uint16_t BUS_ACCESS_RD = 0x60;
constexpr std::uint16_t MEM_ACCESS_RD = 0x66;
constexpr std::uint16_t L3D_CACHE_RD = 0xA0;
constexpr std::uint16_t L3D_CACHE_REFILL_RD = 0xA2;
};  // namespace arm_pmuv3

class PmuEventsPrivate {
  friend class PmuEvents;
  PerfEvent l1d_cache;
  PerfEvent l1d_cache_refill;
  PerfEvent l2d_cache_refill;
  PerfEvent l3d_cache_refill;
  PerfEvent ll_cache_miss;
  PerfEvent cpu_cycles;
  PerfEvent stall_frontend;
  PerfEvent stall_backend;
  PerfEvent br_pred;
  PerfEvent br_mis_pred;
};

PmuEvents::PmuEvents() : priv(new PmuEventsPrivate) {}
PmuEvents::~PmuEvents() { delete priv; }

void PmuEvents::StartRecording() {
  priv->l1d_cache.Start(PERF_TYPE_RAW, arm_pmuv3::L1D_CACHE);
  priv->l1d_cache_refill.Start(PERF_TYPE_RAW, arm_pmuv3::L1D_CACHE_REFILL);
  priv->l2d_cache_refill.Start(PERF_TYPE_RAW, arm_pmuv3::L2D_CACHE_REFILL);
  priv->l3d_cache_refill.Start(PERF_TYPE_RAW, arm_pmuv3::L3D_CACHE_REFILL);
  priv->ll_cache_miss.Start(PERF_TYPE_RAW, arm_pmuv3::LL_CACHE_MISS);
  priv->cpu_cycles.Start(PERF_TYPE_RAW, arm_pmuv3::CPU_CYCLES);
  priv->stall_frontend.Start(PERF_TYPE_RAW, arm_pmuv3::STALL_FRONTEND);
  priv->stall_backend.Start(PERF_TYPE_RAW, arm_pmuv3::STALL_BACKEND);
  priv->br_pred.Start(PERF_TYPE_RAW, arm_pmuv3::BR_PRED);
  priv->br_mis_pred.Start(PERF_TYPE_RAW, arm_pmuv3::BR_MIS_PRED);
}

void PmuEvents::StopRecording() {
  priv->l1d_cache.Stop();
  priv->l1d_cache_refill.Stop();
  priv->l2d_cache_refill.Stop();
  priv->l3d_cache_refill.Stop();
  priv->ll_cache_miss.Stop();
  priv->cpu_cycles.Stop();
  priv->stall_frontend.Stop();
  priv->stall_backend.Stop();
  priv->br_pred.Stop();
  priv->br_mis_pred.Stop();
}

float PmuEvents::BranchMispredictionRate() const {
  std::int64_t br_pred = priv->br_pred.Count();
  std::int64_t br_mis_pred = priv->br_mis_pred.Count();
  return static_cast<float>(br_mis_pred) / br_pred;
}

float PmuEvents::FrontendStallRate() const {
  std::int64_t cpu_cycles = priv->cpu_cycles.Count();
  std::int64_t stall_frontend = priv->stall_frontend.Count();
  return static_cast<float>(stall_frontend) / cpu_cycles;
}

float PmuEvents::BackendStallRate() const {
  std::int64_t cpu_cycles = priv->cpu_cycles.Count();
  std::int64_t stall_backend = priv->stall_backend.Count();
  return static_cast<float>(stall_backend) / cpu_cycles;
}

float PmuEvents::L1AccessCount() const {
  return static_cast<float>(priv->l1d_cache.Count());
}

float PmuEvents::L1RefillCount() const {
  return static_cast<float>(priv->l1d_cache_refill.Count());
}

float PmuEvents::L2RefillCount() const {
  return static_cast<float>(priv->l2d_cache_refill.Count());
}

float PmuEvents::L3RefillCount() const {
  // Important: this was discovered in the context of the above experiments,
  // which also tested the _RD variants of these counters. So it's possible that
  // it's just not needed here with the default (non _RD) counters.
  //
  // Some CPUs implement LL_CACHE_MISS[_RD], some implement
  // L3D_CACHE_REFILL[_RD]. It seems that either one of these two counters is
  // zero, or they roughly both agree with each other. Therefore, taking the max
  // of them is a reasonable way to get something more portable across various
  // CPUs.
  return static_cast<float>(
      std::max(priv->l3d_cache_refill.Count(), priv->ll_cache_miss.Count()));
}

}  // namespace ruy
