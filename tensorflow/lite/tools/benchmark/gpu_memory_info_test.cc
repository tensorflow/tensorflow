/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/benchmark/gpu_memory_info.h"

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif  // defined(__linux__)

#if defined(__APPLE__)
#include <dlfcn.h>

#include <cstddef>
#include <cstring>
#endif  // defined(__APPLE__)

#if defined(_WIN32)
#include <vector>
#endif  // defined(_WIN32)

#if defined(__APPLE__) || defined(_WIN32)
#include <iostream>
#endif  // defined(__APPLE__) || defined(_WIN32)

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tflite {
namespace benchmark {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

constexpr int64_t kKb = 1024;
constexpr int64_t kMb = 1024 * 1024;

TEST(GpuMemorySnapshotTest, DefaultsToUnavailableAndReportsHasGpu) {
  GpuMemorySnapshot snapshot;
  EXPECT_FALSE(snapshot.HasGpu());
  EXPECT_EQ(snapshot.gpu_bytes, GpuMemorySnapshot::kUnavailable);

  snapshot.gpu_bytes = 320 * kMb;
  EXPECT_TRUE(snapshot.HasGpu());
}

TEST(ParseProcFilesTest, ParsesStatusSmapsRollupAndAndroidGpuMappings) {
  const auto vm_rss = internal::ParseProcStatusVmRssBytes(
      "Name:\tbenchmark_model\n"
      "VmRSS:\t  754928 kB\n"
      "RssAnon:\t   53104 kB\n"
      "RssFile:\t  701072 kB\n"
      "RssShmem:\t     752 kB\n");
  ASSERT_TRUE(vm_rss.has_value());
  EXPECT_EQ(*vm_rss, 754928 * kKb);

  const auto smaps_rss = internal::ParseSmapsRollupRssBytes(
      "5e168565c000-7fffd4f64000 ---p 00000000 00:00 0 [rollup]\n"
      "Rss:              198128 kB\n"
      "Pss:               86854 kB\n");
  ASSERT_TRUE(smaps_rss.has_value());
  EXPECT_EQ(*smaps_rss, 198128 * kKb);

  const internal::AndroidGpuMappings mappings =
      internal::ParseProcMapsAndroidGpu(
          "7000000000-7006400000 rw-s 00000000 00:01 10 /kgsl-3d0 (deleted)\n"
          "7006400000-7007800000 rw-s 00000000 00:05 11 /dev/kgsl-3d0\n"
          "7100000000-7112c00000 rw-s 00000000 00:05 12 /dev/dri/renderD128\n"
          "7112c00000-7112d00000 rw-s 00000000 00:05 13 /dev/dri/card0\n"
          "7200000000-7238400000 rw-s 00000000 00:14 14 /dev/mali0\n");
  EXPECT_TRUE(mappings.maps_accessible);
  EXPECT_EQ(mappings.kgsl_bytes, (100 + 20) * kMb);
  EXPECT_EQ(mappings.drm_bytes, 300 * kMb);
  EXPECT_TRUE(mappings.has_mali);
}

TEST(CombineAndroidGpuMemoryTest, ReportsMaliDriverPoolAndDmaBufBreakdown) {
  internal::DmaBufInfo dmabuf;
  dmabuf.fdinfo_accessible = true;
  dmabuf.total_bytes = 64 * kMb;
  dmabuf.by_exporter = {{"system-uncached", 64 * kMb}};

  const GpuMemorySnapshot snapshot = internal::CombineAndroidGpuMemory(
      /*smaps_rss_bytes=*/140 * kMb, /*vm_rss_bytes=*/680 * kMb,
      internal::AndroidGpuMappings{.has_mali = true, .maps_accessible = true},
      dmabuf);

  EXPECT_EQ(snapshot.gpu_bytes, (540 + 64) * kMb);
  EXPECT_THAT(snapshot.gpu_breakdown,
              ElementsAre(Pair("gpu_driver_mali", 540 * kMb),
                          Pair("dmabuf:system-uncached", 64 * kMb)));
}

TEST(CombineAndroidGpuMemoryTest, ReportsAdrenoKgslAndDmaBufBreakdown) {
  const GpuMemorySnapshot adreno = internal::CombineAndroidGpuMemory(
      /*smaps_rss_bytes=*/std::nullopt, /*vm_rss_bytes=*/std::nullopt,
      internal::AndroidGpuMappings{.kgsl_bytes = 320 * kMb,
                                   .maps_accessible = true},
      internal::DmaBufInfo{.total_bytes = 20 * kMb,
                           .by_exporter = {{"qcom,system", 20 * kMb}},
                           .fdinfo_accessible = true});

  EXPECT_EQ(adreno.gpu_bytes, 340 * kMb);
  EXPECT_THAT(adreno.gpu_breakdown,
              ElementsAre(Pair("gpu_driver_kgsl", 320 * kMb),
                          Pair("dmabuf:qcom,system", 20 * kMb)));
}

TEST(CombineAndroidGpuMemoryTest,
     IncludesUnmappedDriverPoolFromBoundaryGpumemOnPowerVr) {
  // PowerVR maps 60 MB on /dev/dri/renderD128, imports a 32 MB DMA-BUF, and
  // holds another 256 MB in unmapped device-only OpenCL buffers reported via
  // dumpsys gpu --gpumem (which includes the 60 MB DRM pool and 32 MB DMA-BUF).
  const GpuMemorySnapshot snapshot = internal::CombineAndroidGpuMemory(
      /*smaps_rss_bytes=*/std::nullopt, /*vm_rss_bytes=*/std::nullopt,
      internal::AndroidGpuMappings{.drm_bytes = 60 * kMb,
                                   .maps_accessible = true},
      internal::DmaBufInfo{.total_bytes = 32 * kMb,
                           .by_exporter = {{"system", 32 * kMb}},
                           .fdinfo_accessible = true},
      /*external_gpumem_bytes=*/(60 + 32 + 256) * kMb);

  EXPECT_EQ(snapshot.gpu_bytes, (60 + 32 + 256) * kMb);
  EXPECT_THAT(snapshot.gpu_breakdown,
              ElementsAre(Pair("gpu_driver_unmapped", 256 * kMb),
                          Pair("gpu_driver_drm", 60 * kMb),
                          Pair("dmabuf:system", 32 * kMb)));
}

TEST(CombineAndroidGpuMemoryTest, ReportsUnavailableWhenAllSourcesRestricted) {
  const GpuMemorySnapshot restricted = internal::CombineAndroidGpuMemory(
      /*smaps_rss_bytes=*/std::nullopt, /*vm_rss_bytes=*/std::nullopt,
      internal::AndroidGpuMappings{.maps_accessible = false},
      internal::DmaBufInfo{.fdinfo_accessible = false},
      /*external_gpumem_bytes=*/GpuMemorySnapshot::kUnavailable);

  EXPECT_FALSE(restricted.HasGpu());
  EXPECT_EQ(restricted.gpu_bytes, GpuMemorySnapshot::kUnavailable);
}

TEST(ParseDumpsysGpuMemPidBytesTest, SumsMatchingPidAcrossGpusAndHandlesZero) {
  constexpr absl::string_view kOutput =
      "Memory snapshot for GPU 0:\n"
      "Global total: 600000000\n"
      "Proc 449 total: 1048576\n"
      "Proc 15671 total: 268435456\n"
      "Memory snapshot for GPU 1:\n"
      "Proc 15671 total: 67108864\n";

  EXPECT_EQ(internal::ParseDumpsysGpuMemPidBytes(kOutput, 15671),
            (256 + 64) * kMb);
  EXPECT_EQ(internal::ParseDumpsysGpuMemPidBytes(kOutput, 99999), 0);
  EXPECT_EQ(internal::ParseDumpsysGpuMemPidBytes("Permission denied", 15671),
            GpuMemorySnapshot::kUnavailable);
}

TEST(CombineLinuxGpuMemoryTest, AggregatesDrmRegionsAndNvml) {
  const auto amd_client = internal::ParseDrmFdinfoClient(
      internal::ParseFdinfoFields("drm-driver:\tamdgpu\n"
                                  "drm-client-id:\t17\n"
                                  "drm-pdev:\t0000:03:00.0\n"
                                  "drm-total-vram:\t1048576 KiB\n"
                                  "drm-resident-vram:\t524288 KiB\n"
                                  "drm-total-gtt:\t128 MiB\n"));
  ASSERT_TRUE(amd_client.has_value());

  const GpuMemorySnapshot snapshot =
      internal::CombineLinuxGpuMemory({*amd_client}, /*fdinfo_accessible=*/true,
                                      /*nvml_process_bytes=*/256 * kMb);

  EXPECT_EQ(snapshot.gpu_bytes, (512 + 128 + 256) * kMb);
  EXPECT_THAT(snapshot.gpu_breakdown, ElementsAre(Pair("drm:vram", 512 * kMb),
                                                  Pair("nvml:vram", 256 * kMb),
                                                  Pair("drm:gtt", 128 * kMb)));

  // Driver without memory keys and without NVML reports unavailable.
  const auto nouveau_client = internal::ParseDrmFdinfoClient(
      internal::ParseFdinfoFields("drm-driver:\tnouveau\ndrm-client-id:\t4\n"));
  ASSERT_TRUE(nouveau_client.has_value());
  const GpuMemorySnapshot unsupported = internal::CombineLinuxGpuMemory(
      {*nouveau_client}, /*fdinfo_accessible=*/true,
      /*nvml_process_bytes=*/std::nullopt);
  EXPECT_FALSE(unsupported.HasGpu());
  EXPECT_THAT(unsupported.gpu_breakdown, IsEmpty());
}

TEST(CombineAppleGpuMemoryTest, CombinesLedgerAndMetalMappedWeights) {
  const GpuMemorySnapshot snapshot = internal::CombineAppleGpuMemory(
      internal::AppleGpuLedger{
          .graphics_bytes = 300 * kMb,
          .graphics_compressed_bytes = 44 * kMb,
          .neural_bytes = 20 * kMb,
          .neural_compressed_bytes = 6 * kMb,
      },
      /*metal_allocated_bytes=*/(370 + 1024) * kMb);

  EXPECT_EQ(snapshot.gpu_bytes, (344 + 26 + 1024) * kMb);
  EXPECT_THAT(
      snapshot.gpu_breakdown,
      ElementsAre(Pair("mapped_weights", 1024 * kMb),
                  Pair("graphics", 344 * kMb), Pair("neural", 26 * kMb)));

  const GpuMemorySnapshot unavailable =
      internal::CombineAppleGpuMemory(std::nullopt, std::nullopt);
  EXPECT_FALSE(unavailable.HasGpu());
  EXPECT_EQ(unavailable.gpu_bytes, GpuMemorySnapshot::kUnavailable);
}

TEST(CombineWindowsGpuMemoryTest, AggregatesLocalAndNonLocalAcrossAdapters) {
  const GpuMemorySnapshot snapshot = internal::CombineWindowsGpuMemory({
      {.description = "NVIDIA GeForce RTX 4070",
       .local_bytes = 1000 * kMb,
       .non_local_bytes = 200 * kMb},
      {.description = "Intel(R) UHD Graphics",
       .local_bytes = 0,
       .non_local_bytes = 50 * kMb},
      {.description = "Microsoft Basic Render Driver",
       .local_bytes = 0,
       .non_local_bytes = 0},
  });

  EXPECT_EQ(snapshot.gpu_bytes, 1250 * kMb);
  EXPECT_THAT(snapshot.gpu_breakdown,
              ElementsAre(Pair("dxgi:local", 1000 * kMb),
                          Pair("dxgi:non_local", 250 * kMb)));

  const GpuMemorySnapshot unavailable = internal::CombineWindowsGpuMemory({});
  EXPECT_FALSE(unavailable.HasGpu());
  EXPECT_EQ(unavailable.gpu_bytes, GpuMemorySnapshot::kUnavailable);
}

#if defined(__linux__)
TEST(ReadProcGpuMemorySnapshotTest, DeduplicatesDmaBufsAndDrmClientsInFakeDir) {
  const std::string proc_dir = absl::StrCat(
      testing::TempDir(), "/fake_proc_self_", static_cast<int>(getpid()));
  const std::string fdinfo_dir = absl::StrCat(proc_dir, "/fdinfo");
  ASSERT_EQ(mkdir(proc_dir.c_str(), 0755), 0);
  ASSERT_EQ(mkdir(fdinfo_dir.c_str(), 0755), 0);

  auto write_file = [](const std::string& path, absl::string_view text) {
    std::ofstream out(path);
    out << text;
  };

  write_file(absl::StrCat(proc_dir, "/status"),
             "VmRSS:\t 307200 kB\nRssShmem:\t 10240 kB\n");
  write_file(absl::StrCat(proc_dir, "/smaps_rollup"), "Rss:\t 102400 kB\n");
  write_file(absl::StrCat(proc_dir, "/maps"),
             "7000000000-700c800000 rw-s 00000000 00:14 14 /dev/mali0\n");

  // Duplicate DMA-BUF descriptors pointing to the same inode 42 must be counted
  // once.
  constexpr absl::string_view kDmaBufFdinfo =
      "pos:\t0\nino:\t42\nsize:\t33554432\nexp_name:\tsystem\n";
  write_file(absl::StrCat(fdinfo_dir, "/3"), kDmaBufFdinfo);
  write_file(absl::StrCat(fdinfo_dir, "/4"), kDmaBufFdinfo);

  const GpuMemorySnapshot android_snap =
      internal::ReadAndroidProcGpuMemorySnapshot(proc_dir);
  EXPECT_EQ(android_snap.gpu_bytes, (200 + 32) * kMb);

  // Duplicate DRM descriptors for the same client id 7 must also be counted
  // once.
  constexpr absl::string_view kDrmFdinfo =
      "drm-driver:\ti915\ndrm-pdev:\t0000:00:02.0\ndrm-client-id:\t7\n"
      "drm-total-system0:\t64 MiB\n";
  write_file(absl::StrCat(fdinfo_dir, "/5"), kDrmFdinfo);
  write_file(absl::StrCat(fdinfo_dir, "/6"), kDrmFdinfo);

  const GpuMemorySnapshot linux_snap = internal::ReadLinuxProcGpuMemorySnapshot(
      proc_dir, /*nvml_process_bytes=*/std::nullopt);
  EXPECT_EQ(linux_snap.gpu_bytes, 64 * kMb);

  unlink(absl::StrCat(fdinfo_dir, "/3").c_str());
  unlink(absl::StrCat(fdinfo_dir, "/4").c_str());
  unlink(absl::StrCat(fdinfo_dir, "/5").c_str());
  unlink(absl::StrCat(fdinfo_dir, "/6").c_str());
  rmdir(fdinfo_dir.c_str());
  unlink(absl::StrCat(proc_dir, "/status").c_str());
  unlink(absl::StrCat(proc_dir, "/smaps_rollup").c_str());
  unlink(absl::StrCat(proc_dir, "/maps").c_str());
  rmdir(proc_dir.c_str());
}

TEST(ReadProcGpuMemorySnapshotTest, LiveProcessSnapshotDoesNotCrash) {
  ASSERT_TRUE(GpuMemorySnapshot::IsSupported());
  const GpuMemorySnapshot snap =
      GetGpuMemorySnapshot(/*include_external_tools=*/false);
  EXPECT_GE(snap.gpu_bytes, GpuMemorySnapshot::kUnavailable);
}
#endif  // defined(__linux__)

#if defined(__APPLE__)
TEST(ReadAppleGpuMemorySnapshotTest, LiveLedgerAndMetalAllocationAreTracked) {
  ASSERT_TRUE(GpuMemorySnapshot::IsSupported());
  const std::optional<internal::AppleGpuLedger> ledger_before =
      internal::ReadAppleGpuLedger();
  ASSERT_TRUE(ledger_before.has_value());

  const GpuMemorySnapshot before =
      GetGpuMemorySnapshot(/*include_external_tools=*/false);
  ASSERT_TRUE(before.HasGpu());

  void* objc_lib = dlopen("/usr/lib/libobjc.A.dylib", RTLD_LAZY | RTLD_LOCAL);
  void* metal_lib = dlopen("/System/Library/Frameworks/Metal.framework/Metal",
                           RTLD_LAZY | RTLD_LOCAL);
  ASSERT_NE(objc_lib, nullptr);
  ASSERT_NE(metal_lib, nullptr);

  using Sel = void*;
  using RegisterNameFn = Sel (*)(const char*);
  using CreateDefaultDeviceFn = void* (*)();
  using NewBufferFn = void* (*)(void*, Sel, size_t, size_t);
  using MsgSendPtrFn = void* (*)(void*, Sel);
  using MsgSendVoidFn = void (*)(void*, Sel);

  auto register_name =
      reinterpret_cast<RegisterNameFn>(dlsym(objc_lib, "sel_registerName"));
  void* msg_send = dlsym(objc_lib, "objc_msgSend");
  auto create_device = reinterpret_cast<CreateDefaultDeviceFn>(
      dlsym(metal_lib, "MTLCreateSystemDefaultDevice"));
  ASSERT_NE(register_name, nullptr);
  ASSERT_NE(msg_send, nullptr);
  ASSERT_NE(create_device, nullptr);

  void* device = create_device();
  if (device == nullptr) {
    std::cerr << "No Metal device available on this Mac runner; "
                 "verified task_info ledger only (before.gpu_bytes="
              << before.gpu_bytes << ").\n";
    return;
  }

  const std::optional<int64_t> metal_before =
      internal::QueryAppleMetalAllocatedBytes();
  ASSERT_TRUE(metal_before.has_value());

  constexpr size_t kAllocBytes = 16 * 1024 * 1024;
  constexpr size_t kStorageModeShared = 0;
  auto new_buffer = reinterpret_cast<NewBufferFn>(msg_send);
  void* buffer =
      new_buffer(device, register_name("newBufferWithLength:options:"),
                 kAllocBytes, kStorageModeShared);
  ASSERT_NE(buffer, nullptr);

  auto get_contents = reinterpret_cast<MsgSendPtrFn>(msg_send);
  void* contents = get_contents(buffer, register_name("contents"));
  ASSERT_NE(contents, nullptr);
  std::memset(contents, 1, kAllocBytes);

  const GpuMemorySnapshot after =
      GetGpuMemorySnapshot(/*include_external_tools=*/false);
  const std::optional<int64_t> metal_after =
      internal::QueryAppleMetalAllocatedBytes();
  ASSERT_TRUE(metal_after.has_value());
  std::cerr << "Apple GPU memory before=" << before.gpu_bytes
            << " after=" << after.gpu_bytes << " metal_before=" << *metal_before
            << " metal_after=" << *metal_after << "\n";
  EXPECT_GE(*metal_after, *metal_before + 8 * kMb);
  EXPECT_GE(after.gpu_bytes, before.gpu_bytes + 8 * kMb);

  auto release_obj = reinterpret_cast<MsgSendVoidFn>(msg_send);
  release_obj(buffer, register_name("release"));
  release_obj(device, register_name("release"));
}
#endif  // defined(__APPLE__)

#if defined(_WIN32)
TEST(ReadWindowsGpuMemorySnapshotTest, LiveDxgiAdaptersAreQueryable) {
  ASSERT_TRUE(GpuMemorySnapshot::IsSupported());

  const std::vector<internal::WindowsAdapterVideoMemory> adapters =
      internal::ReadWindowsAdapterVideoMemory();
  ASSERT_FALSE(adapters.empty());
  for (const auto& adapter : adapters) {
    std::cerr << "Windows DXGI adapter: " << adapter.description
              << " local=" << adapter.local_bytes
              << " non_local=" << adapter.non_local_bytes << "\n";
  }

  const GpuMemorySnapshot snap =
      GetGpuMemorySnapshot(/*include_external_tools=*/false);
  EXPECT_TRUE(snap.HasGpu());
  EXPECT_GE(snap.gpu_bytes, 0);
}
#endif  // defined(_WIN32)

}  // namespace
}  // namespace benchmark
}  // namespace tflite
