/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/rocm_rocdl_path.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/env_var.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/random.h"
#include "tsl/profiler/lib/traceme.h"

#ifdef HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY
#include <array>

#include "absl/base/const_init.h"
#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(elf)
#endif

#ifdef HAS_SUPPORT_FOR_EMBEDDED_LIB_DEVICE
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_device_lib_data.h"
#else
constexpr const char* kAMDGPUDeviceLibData = "";
#endif

namespace xla {
namespace gpu {
namespace {

// Inline threshold value to use in LLVM AMDGPU backend.
const int kAMDGPUInlineThreshold = 0x100000;
const int32_t kAMDGPUAbiVersion = 500;

// Gets the ROCm-Device-Libs filenames for a particular AMDGPU version.
std::vector<std::string> GetROCDLPaths(const std::string& rocdl_dir_path) {
  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  result.reserve(2);
  for (absl::string_view filename : {"ocml.bc", "ockl.bc"}) {
    result.emplace_back(tsl::io::JoinPath(rocdl_dir_path, filename));
  }

  return result;
}

struct HsacoCache {
  using HashType = std::array<uint8_t, 32>;

 private:
  struct Hash64 {
    size_t operator()(const HashType& s) const noexcept {
      return *reinterpret_cast<const size_t*>(s.data());
    }
  };

  absl::Mutex mutex_;
  absl::flat_hash_map<HashType, amdgpu::HsacoResult, Hash64> hsaco_cache_
      ABSL_GUARDED_BY(mutex_);
  std::atomic_int request_count_, hit_count_;
  std::string hsaco_cache_dir_;
  int64_t bitcode_size_threshold_;
  bool keep_temp_files_;

  HsacoCache() {
    auto* env = tsl::Env::Default();
    TF_CHECK_OK(tsl::ReadStringFromEnvVar("TF_XLA_HSACO_CACHE_DIR", "",
                                          &hsaco_cache_dir_));
    // minimal size of llvm Module bitcode to use file cache
    TF_CHECK_OK(tsl::ReadInt64FromEnvVar("TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD",
                                         /*default_val=*/65536,
                                         &bitcode_size_threshold_));

    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_KEEP_XLA_TEMPFILES",
                                        /*default_val=*/false,
                                        &keep_temp_files_));

    if (hsaco_cache_dir_.empty()) {
      LOG(INFO)
          << "TF_XLA_HSACO_CACHE_DIR is not set: HSACO file cache is disabled!";
      return;
    }
    if (!env->IsDirectory(hsaco_cache_dir_).ok()) {
      TF_CHECK_OK(env->CreateDir(hsaco_cache_dir_));
    }
    LOG(INFO) << "HSACO file cache in '" << hsaco_cache_dir_
              << "' is enabled for LLVM modules with bitcode size >= "
              << bitcode_size_threshold_ << " bytes";

    if (hsaco_cache_dir_.back() != '/') hsaco_cache_dir_ += '/';
  }

 public:
  static HsacoCache& i() {
    static HsacoCache obj;
    return obj;
  }

  bool KeepTempFiles() const { return keep_temp_files_; }

  std::string HsacoFilePath(const std::string& hash_str) const {
    return hsaco_cache_dir_ + hash_str + ".hsaco";
  }

  bool Find(const HashType& hash_val, int64_t bitcode_size,
            std::string* hash_str, amdgpu::HsacoResult* result) {
    bool hit = false;
    request_count_++;
    {
      absl::MutexLock lock(&mutex_);
      if (auto it = hsaco_cache_.find(hash_val); it != hsaco_cache_.end()) {
        hit = true, *result = it->second;
      }
    }
    absl::string_view hview(reinterpret_cast<const char*>(hash_val.data()),
                            hash_val.size());
    *hash_str = absl::BytesToHexString(hview);
    if (!hit && !hsaco_cache_dir_.empty() &&
        bitcode_size >= bitcode_size_threshold_) {
      auto hsaco_src_path = HsacoFilePath(*hash_str);
      if (ReadFromFile(hsaco_src_path, &result->hsaco)) {
        hit = true;
        VLOG(1) << "HSACO file cache hit";
      }
    }
    if (hit) hit_count_++;
    VLOG(1) << "HSACO cache: " << request_count_ << " requests, " << hit_count_
            << " hits";
    return hit;
  }

  // attempts to read an hsaco binary file, adds it to in-memory cache, and
  // (if enabled) moves/copies the binary file to the cached location
  bool ReadFromFile(const std::string& hsaco_src_path,
                    std::vector<uint8_t>* hsaco) {
    std::ifstream ifs(hsaco_src_path, std::ios::binary | std::ios::ate);
    size_t fsize = ifs.tellg();
    if (!ifs.is_open() || fsize == 0) return false;
    hsaco->resize(fsize);
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(hsaco->data()), fsize);
    return true;
  }

  void Insert(const HashType& hash_val, int64_t bitcode_size,
              const std::string& src_path, const std::string& tgt_path,
              const amdgpu::HsacoResult& result) {
    absl::MutexLock lock(&mutex_);
    hsaco_cache_.emplace(hash_val, result);

    if (!hsaco_cache_dir_.empty() && bitcode_size >= bitcode_size_threshold_) {
      // write hsaco file to the new location if simple rename fails
      if (!tsl::Env::Default()->RenameFile(src_path, tgt_path).ok()) {
        std::ofstream ofs(tgt_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(result.hsaco.data()),
                  result.hsaco.size());
        std::remove(src_path.c_str());  // remove temporary file
        if (ofs.fail()) {
          LOG(FATAL) << "Unable to write hsaco file cache: " << tgt_path;
        }
      }
    }
  }
};  // HsacoCache

// Per-kernel register spilling and stack information from HSACO metadata.
struct KernelSpillInfo {
  std::string name;
  uint64_t sgpr_spill_count = 0;
  uint64_t vgpr_spill_count = 0;
  uint64_t private_segment_size = 0;
  bool uses_dynamic_stack = false;

  bool HasSpilling() const {
    return sgpr_spill_count > 0 || vgpr_spill_count > 0;
  }

  bool HasStackUsage() const {
    return private_segment_size > 0 || uses_dynamic_stack;
  }
};

// Aggregated register spilling information across all kernels in a module.
struct RegisterSpillInfo {
  std::vector<KernelSpillInfo> kernels;

  bool HasSpilling() const {
    return absl::c_any_of(
        kernels, [](const KernelSpillInfo& k) { return k.HasSpilling(); });
  }

  bool HasStackUsage() const {
    return absl::c_any_of(
        kernels, [](const KernelSpillInfo& k) { return k.HasStackUsage(); });
  }

  // Convert to ModuleStats format for autotuner filtering.
  //
  // Only kernels with private_segment_size > 0 (i.e. spills that use
  // stack scratch memory) are included. SGPR spills that are saved to
  // VGPRs rather than memory do not increase private_segment_size and are
  // not reported here. This means that neither the autotuner's filter
  // (xla_gpu_filter_kernels_spilling_registers_on_autotuning) nor the
  // hard-fail path (xla_gpu_fail_ptx_compilation_on_register_spilling) will
  // discard kernels whose SGPR spills stay in VGPRs. The only difference
  // between these two paths is that the hard-fail path also discards
  // kernels with dynamic stack usage, via HasStackUsage() in
  // EmitModuleToHsaco.
  //
  // AMD metadata does not distinguish load vs store spill bytes, so we use
  // private_segment_size as a conservative proxy for both fields. The
  // autotuner only checks whether the values are > 0, so the exact
  // magnitude does not affect filtering behavior.
  ModuleStats ToModuleStats() const {
    ModuleStats stats;
    for (const KernelSpillInfo& k : kernels) {
      if (k.private_segment_size > 0) {
        KernelStats ks;
        ks.store_bytes_spilled = static_cast<int>(std::min<uint64_t>(
            k.private_segment_size, std::numeric_limits<int>::max()));
        ks.load_bytes_spilled = ks.store_bytes_spilled;
        stats[k.name] = ks;
      }
    }
    return stats;
  }
};

// Parse NT_AMDGPU_METADATA note contents and extract register spill counts.
// The metadata is in MessagePack format containing kernel information.
RegisterSpillInfo ParseAMDGPUMetadataForSpills(llvm::StringRef metadata) {
  // Parse the MsgPack metadata
  llvm::msgpack::Document doc;
  if (!doc.readFromBlob(metadata, /*Multi=*/false)) {
    VLOG(2) << "Could not parse MsgPack metadata from NT_AMDGPU_METADATA note";
    return RegisterSpillInfo{};
  }

  llvm::msgpack::DocNode root = doc.getRoot();
  if (!root.isMap()) {
    VLOG(2) << "AMDGPU metadata root is not a map (unexpected format)";
    return RegisterSpillInfo{};
  }

  // Look for "amdhsa.kernels" array
  llvm::msgpack::MapDocNode root_map = root.getMap();
  auto kernels_it = root_map.find("amdhsa.kernels");

  if (kernels_it == root_map.end() || !kernels_it->second.isArray()) {
    VLOG(2) << "NT_AMDGPU_METADATA found but missing 'amdhsa.kernels' array";
    return RegisterSpillInfo{};
  }

  llvm::msgpack::ArrayDocNode kernels_array = kernels_it->second.getArray();

  RegisterSpillInfo spill_info;
  // Iterate through each kernel, collecting per-kernel spill info.
  for (auto& kernel_node : kernels_array) {
    if (!kernel_node.isMap()) continue;

    llvm::msgpack::MapDocNode kernel_map = kernel_node.getMap();
    KernelSpillInfo kernel_info;

    // Look for ".sgpr_spill_count"
    auto sgpr_it = kernel_map.find(".sgpr_spill_count");
    if (sgpr_it != kernel_map.end() &&
        sgpr_it->second.getKind() == llvm::msgpack::Type::UInt) {
      kernel_info.sgpr_spill_count = sgpr_it->second.getUInt();
    }

    // Look for ".vgpr_spill_count"
    auto vgpr_it = kernel_map.find(".vgpr_spill_count");
    if (vgpr_it != kernel_map.end() &&
        vgpr_it->second.getKind() == llvm::msgpack::Type::UInt) {
      kernel_info.vgpr_spill_count = vgpr_it->second.getUInt();
    }

    // Look for ".private_segment_fixed_size"
    auto priv_it = kernel_map.find(".private_segment_fixed_size");
    if (priv_it != kernel_map.end() &&
        priv_it->second.getKind() == llvm::msgpack::Type::UInt) {
      kernel_info.private_segment_size = priv_it->second.getUInt();
    }

    // Look for ".uses_dynamic_stack"
    auto dyn_it = kernel_map.find(".uses_dynamic_stack");
    if (dyn_it != kernel_map.end() &&
        dyn_it->second.getKind() == llvm::msgpack::Type::Boolean) {
      kernel_info.uses_dynamic_stack = dyn_it->second.getBool();
    }

    // Get kernel name
    auto name_it = kernel_map.find(".name");
    if (name_it != kernel_map.end() &&
        name_it->second.getKind() == llvm::msgpack::Type::String) {
      kernel_info.name = name_it->second.getString().str();
    } else {
      kernel_info.name = "unknown";
    }

    // Log per-kernel spill information with register usage
    if (kernel_info.HasSpilling()) {
      uint64_t kernel_sgpr_count = 0;
      uint64_t kernel_vgpr_count = 0;
      // Look for ".sgpr_count" (total SGPRs used)
      auto sgpr_count_it = kernel_map.find(".sgpr_count");
      if (sgpr_count_it != kernel_map.end() &&
          sgpr_count_it->second.getKind() == llvm::msgpack::Type::UInt) {
        kernel_sgpr_count = sgpr_count_it->second.getUInt();
      }

      // Look for ".vgpr_count" (total VGPRs used)
      auto vgpr_count_it = kernel_map.find(".vgpr_count");
      if (vgpr_count_it != kernel_map.end() &&
          vgpr_count_it->second.getKind() == llvm::msgpack::Type::UInt) {
        kernel_vgpr_count = vgpr_count_it->second.getUInt();
      }

      VLOG(2) << "Kernel '" << kernel_info.name << "' has register spilling: "
              << "SGPR=" << kernel_info.sgpr_spill_count
              << ", VGPR=" << kernel_info.vgpr_spill_count
              << ". Register count: SGPR=" << kernel_sgpr_count
              << ", VGPR=" << kernel_vgpr_count;
    }

    // Log per-kernel stack usage
    if (kernel_info.HasStackUsage()) {
      VLOG(2) << "Kernel '" << kernel_info.name << "' stack usage: "
              << "private=" << kernel_info.private_segment_size << ", dynamic="
              << (kernel_info.uses_dynamic_stack ? "true" : "false");
    }

    spill_info.kernels.push_back(std::move(kernel_info));
  }

  return spill_info;
}

// ELF note descriptor alignment per ELF specification
constexpr int kElfNoteDescAlignment = 4;

// Returns spill counts by parsing AMDGPU metadata from note sections of HSACO
// ELF binary.
//
// HSACO file (ELF binary)
//   -- .note section(s)
//       -- ELF Note with type=NT_AMDGPU_METADATA
//           -- MessagePack data
//               -- Root map
//                   -- "amdhsa.kernels" array
//                       -- Each kernel object
//                           - ".sgpr_spill_count"
//                           - ".vgpr_spill_count"
//                           - ... (other kernel properties)
RegisterSpillInfo ExtractRegisterSpillingFromHsaco(
    const std::vector<uint8_t>& hsaco) {
  // Create memory buffer from HSACO data
  std::unique_ptr<llvm::MemoryBuffer> mem_buffer =
      llvm::MemoryBuffer::getMemBuffer(
          llvm::StringRef(reinterpret_cast<const char*>(hsaco.data()),
                          hsaco.size()),
          "", /*RequiresNullTerminator=*/false);

  // Parse as ELF object file
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_or_err =
      llvm::object::ObjectFile::createObjectFile(mem_buffer->getMemBufferRef());

  if (!obj_or_err) {
    VLOG(2) << "Could not parse HSACO as ELF object file: "
            << llvm::toString(obj_or_err.takeError());
    return RegisterSpillInfo{};
  }

  llvm::object::ObjectFile* obj = obj_or_err->get();

  // Cast to ELF64LE object file (AMDGPU uses 64-bit little-endian ELF)
  auto* elf_obj = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(obj);
  if (!elf_obj) {
    VLOG(2) << "HSACO is not a 64-bit little-endian ELF file";
    return RegisterSpillInfo{};
  }

  // Get the underlying ELFFile to access the notes() API
  const auto& elf_file = elf_obj->getELFFile();

  for (const auto& section : elf_obj->sections()) {
    llvm::Expected<const typename llvm::object::ELF64LEObjectFile::Elf_Shdr*>
        shdr_or_err = elf_obj->getSection(section.getRawDataRefImpl());

    if (!shdr_or_err) {
      continue;  // Skip sections we can't access
    }

    const auto* shdr = *shdr_or_err;

    if (shdr->sh_type != llvm::ELF::SHT_NOTE) {
      continue;
    }

    llvm::Error err = llvm::Error::success();
    for (const auto& note : elf_file.notes(*shdr, err)) {
      if (note.getType() == llvm::ELF::NT_AMDGPU_METADATA) {
        llvm::StringRef metadata =
            note.getDescAsStringRef(kElfNoteDescAlignment);

        if (metadata.empty()) {
          VLOG(2) << "Found NT_AMDGPU_METADATA note but it contains no data";
          continue;
        }

        // Parse the metadata and extract spill counts, return immediately
        return ParseAMDGPUMetadataForSpills(metadata);
      }
    }

    if (err) {
      VLOG(2) << "Error parsing notes: " << llvm::toString(std::move(err));
    }
  }

  // If we reach here, no metadata was found
  VLOG(2) << "No AMDGPU metadata found in HSACO";
  return RegisterSpillInfo{};
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
absl::StatusOr<std::string> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine,
    const DebugOptions& debug_options) {
  auto* env = tsl::Env::Default();
  std::vector<std::string> tempdir_vector;
  env->GetLocalTempDirectories(&tempdir_vector);
  if (tempdir_vector.empty()) {
    return xla::Internal(
        "Unable to locate a temporary directory for compile-time artifacts.");
  }
  std::string tempdir_name = tempdir_vector.front();
  VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;

  // Prepare filenames for all stages of compilation:
  // IR, binary ISA, and HSACO.
  std::string random_number = std::to_string(tsl::random::New64());
  absl::string_view module_id = module->getModuleIdentifier();
  auto gen_path = [module_id, &random_number,
                   &tempdir_name](absl::string_view ext) {
    return tsl::io::JoinPath(tempdir_name,
                             absl::StrCat(module_id, random_number, ext));
  };

  std::string ir_path = gen_path(".ll"), ir_opt_path = gen_path("_opt.ll"),
              isabin_path = gen_path(".o"), hsaco_path = gen_path(".hsaco");

  absl::Cleanup cleanup = [&] {
    if (!HsacoCache::i().KeepTempFiles()) {
      std::remove(ir_path.c_str());
      std::remove(isabin_path.c_str());
      std::remove(ir_opt_path.c_str());
    }
  };

  std::error_code ec;
  {  // Dump LLVM IR.
    llvm::raw_fd_ostream ir_fs(ir_path, ec, llvm::sys::fs::OF_None);
    module->print(ir_fs, nullptr);
  }

  {  // Emit GCN ISA binary.
    llvm::legacy::PassManager pm;
    pm.add(new llvm::TargetLibraryInfoWrapperPass(
        llvm::Triple(module->getTargetTriple())));

    llvm::raw_fd_ostream isabin_fs(isabin_path, ec, llvm::sys::fs::OF_Text);
    module->setDataLayout(target_machine->createDataLayout());
    target_machine->addPassesToEmitFile(pm, isabin_fs, nullptr,
                                        llvm::CodeGenFileType::ObjectFile);
    pm.run(*module);
  }

  if (HsacoCache::i().KeepTempFiles()) {
    llvm::raw_fd_ostream ir_fs(ir_opt_path, ec, llvm::sys::fs::OF_None);
    module->print(ir_fs, nullptr);
  }

  if (debug_options.xla_gpu_use_inprocess_lld()) {
#ifdef HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY
    static absl::Mutex lld_mu(absl::kConstInit);

    std::initializer_list<const char*> args{
        "ld.lld",           "--threads=1",       "-shared",
        "--no-undefined",   isabin_path.c_str(), "-o",
        hsaco_path.c_str(),
    };

    std::string error_message;
    llvm::raw_string_ostream os(error_message);
    lld::Result result;
    {
      absl::MutexLock lock(&lld_mu);
      result =
          lld::lldMain(args, llvm::nulls(), os, {{lld::Gnu, &lld::elf::link}});
    }
    CHECK(result.canRunAgain)
        << "ld.lld (in-process) failed with fatal error " << error_message;
    if (result.retCode) {
      return xla::Internal(
          "ld.lld (in-process) execute fail: %s, error code %d", error_message,
          result.retCode);
    }
#else
    CHECK(false) << "Inprocess LLD is not supported.";
#endif  // HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY
  } else {
    // Locate lld.
    std::string lld_path;
    if (std::getenv("LLVM_PATH")) {
      lld_path = tsl::io::JoinPath(std::getenv("LLVM_PATH"), "bin");
    } else {
      lld_path = tsl::io::JoinPath(tsl::RocmRoot(), "llvm/bin");
    }
    auto lld_program = llvm::sys::findProgramByName("ld.lld", {lld_path});
    if (!lld_program) {
      return xla::Internal("unable to find ld.lld in PATH: %s",
                           lld_program.getError().message());
    }
    std::initializer_list<llvm::StringRef> lld_args{
        "ld.lld",         "-flavor",   "gnu", "-shared",
        "--no-undefined", isabin_path, "-o",  hsaco_path,
    };

    std::string error_message;
    int lld_result = llvm::sys::ExecuteAndWait(
        *lld_program, lld_args, std::nullopt, {}, 0, 0, &error_message);
    if (lld_result) {
      return xla::Internal("ld.lld execute fail: %s, error code %d",
                           error_message, lld_result);
    }
  }  // xla_gpu_use_inprocess_lld

  return hsaco_path;
}

absl::Status AMDGPUTargetModuleLinker(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& device_bitcode_dir_path) {
  // Link the input module with ROCDL.

  auto compute_capability = gpu_version.rocm_compute_capability();
  if (!compute_capability) {
    return xla::Internal("Incompatible compute capability was specified.");
  }

  TF_RETURN_IF_ERROR(
      amdgpu::LinkROCDLIfNecessary(module, compute_capability->gfx_version(),
                                   debug_options, device_bitcode_dir_path));

  // If ftz is enabled, set it as an attribute on every function in the module.
  if (debug_options.xla_gpu_ftz()) {
    for (llvm::Function& fn : *module) {
      fn.addFnAttr("denormal-fp-math-f32", "preserve-sign");
    }
  }
  module->addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
                        kAMDGPUAbiVersion);

  return absl::OkStatus();
}

// The following routine maps a feature token extracted from the
// hipDeviceProp_t::gcnArchName string, and maps it to a valid feature_str
// to be used for creating the AMDGPUTarget.
// This mapping is currently in a state of flux because TF XLA uses its
// own copy of LLVM, which is different from the LLVM version used by
// hipcc/runtime in the ROCm install. Ordinarily this is not a problem,
// but right now, the LLVM version used by hipcc/runtime has "targetID"
// related changes which have not yet been upstreamed (to the LLVM repo)
// When that upstreaming happens (and TF LLVM pointer moves past the
// upstream commit), the following mapping will need to change
std::string MapGCNArchNameTokenToFeatureStr(const std::string& token,
                                            const std::string& gfx) {
  if (token == "sramecc+") {
    return "+sramecc";
  }
  if (token == "sramecc-") {
    if (gfx == "gfx90a" || gfx == "gfx942") {
      return "";
    }
    return "-sramecc";
  }
  if (token == "xnack+") {
    return "+xnack";
  }
  if (token == "xnack-") {
    return "-xnack";
  }
  return "";
}

std::pair<std::string, std::string> GetFeatureStrFromGCNArchName(
    const std::string& gcn_arch_name) {
  std::string gfx = gcn_arch_name;
  // For ROCm versions 4.0 and greater, we need to specify the correct
  // feature str, based on the underlying GPU HW to get max performance.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  if (!tokens.empty()) gfx = tokens[0];

  std::string mapped_tokens;
  for (size_t i = 1; i < tokens.size(); i++) {
    // Skip the first token, that is the gfxNNN str
    // The rest of the tokens are the feature/targetid strings
    auto mapped_token = MapGCNArchNameTokenToFeatureStr(tokens[i], gfx);
    if (!mapped_token.empty()) {
      mapped_tokens += "," + mapped_token;
    }
  }
  return std::pair{gfx, mapped_tokens};
}

// Returns the directory containing ROCm-Device-Libs files.
std::string GetROCDLDir(const DebugOptions& debug_options) {
  std::vector<std::string> potential_rocdl_dirs;
  const std::string& datadir = debug_options.xla_gpu_cuda_data_dir();
  if (!datadir.empty()) {
    potential_rocdl_dirs.push_back(datadir);
  }
  potential_rocdl_dirs.push_back(tsl::RocdlRoot());

  // Tries all potential ROCDL directories in the order they are inserted.
  // Returns the first directory that exists in the file system.
  for (const std::string& potential_rocdl_dir : potential_rocdl_dirs) {
    if (tsl::Env::Default()->IsDirectory(potential_rocdl_dir).ok()) {
      VLOG(2) << "Found ROCm-Device-Libs dir " << potential_rocdl_dir;
      return potential_rocdl_dir;
    }
    VLOG(2) << "Unable to find potential ROCm-Device-Libs dir "
            << potential_rocdl_dir;
  }

  // Last resort: maybe in the current folder.
  return ".";
}

void AMDGPUBackendInit(const DebugOptions& debug_options,
                       std::string& rocdl_dir_path) {
  // Initialize the AMDGPU target; it's the only target we link with, so call
  // its specific initialization functions instead of the catch-all
  // InitializeAll*.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  rocdl_dir_path = GetROCDLDir(debug_options);
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  gpu::InitializePasses(registry);
}

absl::StatusOr<amdgpu::HsacoResult> CompileToHsacoInternal(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, std::string* hsaco_temp_path) {
  static absl::once_flag backend_init_flag;
  // TODO(rocm) Ideally this would be refreshed if xla_gpu_cuda_data_dir
  // changes.
  static std::string rocdl_dir_path;  // NOLINT: static/global vars forbidden
  absl::call_once(backend_init_flag, AMDGPUBackendInit, debug_options,
                  std::ref(rocdl_dir_path));

  auto cc = gpu_version.rocm_compute_capability();
  if (!cc) {
    return xla::Internal("Incompatible compute capability was specified.");
  }
  llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
  // Construct LLVM TargetMachine for AMDGPU.
  auto [gfx, feature_str] = GetFeatureStrFromGCNArchName(cc->gcn_arch_name());
  auto target_machine =
      GetTargetMachine(default_target_triple, gfx, debug_options, feature_str);

  // Link with ROCm-Device-Libs, and optimize the LLVM module.
  TF_RETURN_IF_ERROR(gpu::LinkAndOptimizeModule(
      module, gpu_version, debug_options, rocdl_dir_path,
      AMDGPUTargetModuleLinker, default_target_triple, target_machine.get(),
      kAMDGPUInlineThreshold));

  // Lower optimized LLVM module to HSA code object.
  TF_ASSIGN_OR_RETURN(
      std::string hsaco_path,
      EmitModuleToHsaco(module, target_machine.get(), debug_options));

  // Check for register spilling using HSACO metadata
  VLOG(2) << "Checking for register spilling in: "
          << module->getModuleIdentifier();

  std::vector<uint8_t> hsaco;
  if (!HsacoCache::i().ReadFromFile(hsaco_path, &hsaco)) {
    return xla::Internal("Unable to read hsaco output file");
  }
  if (hsaco_temp_path) *hsaco_temp_path = std::move(hsaco_path);

  RegisterSpillInfo spill_info = ExtractRegisterSpillingFromHsaco(hsaco);
  if (spill_info.HasSpilling()) {
    // We can have SGPR spills without stack being used. They are saved to
    // VGPRs. In that case, we don't want to discard such kernel, so just
    // report such cases.
    for (const KernelSpillInfo& k : spill_info.kernels) {
      if (k.HasSpilling()) {
        VLOG(1) << "Register spilling in kernel '" << k.name
                << "' (SGPR: " << k.sgpr_spill_count
                << ", VGPR: " << k.vgpr_spill_count << ") in "
                << module->getModuleIdentifier();
      }
    }
  } else {
    VLOG(2) << "No register spilling detected in "
            << module->getModuleIdentifier();
  }

  if (spill_info.HasStackUsage()) {
    for (const KernelSpillInfo& k : spill_info.kernels) {
      if (k.HasStackUsage()) {
        VLOG(1) << "Stack usage in kernel '" << k.name
                << "' (private: " << k.private_segment_size
                << ", dynamic: " << (k.uses_dynamic_stack ? "true" : "false")
                << ") in " << module->getModuleIdentifier();
      }
    }

    // Filter out kernels with register spilling during autotuning
    // This matches NVIDIA's behavior in ptx_compiler_impl.cc
    // TODO: remove ptx from xla_gpu_fail_ptx_compilation_on_register_spilling
    // to make the flag more general
    if (debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling()) {
      VLOG(0) << "Discard module " << module->getModuleIdentifier()
              << " due register spilling or stack usage";
      return xla::Cancelled(
          "Compilation result discarded due to register spilling or stack "
          "usage");
    }
  } else {
    VLOG(2) << "No stack usage detected in " << module->getModuleIdentifier();
  }

  return amdgpu::HsacoResult{std::move(hsaco), spill_info.ToModuleStats()};
}

class sha256_ostream : public llvm::raw_ostream {
  llvm::SHA256& obj_;
  uint64_t pos_ = 0;

  void write_impl(const char* ptr, size_t size) override {
    obj_.update(llvm::StringRef(ptr, size));
    pos_ += size;
  }

  /// Return the current position within the stream.
  uint64_t current_pos() const override { return pos_; }

  void anchor() override {}

  size_t preferred_buffer_size() const override {
    return llvm::raw_ostream::preferred_buffer_size();  // TODO ?
  }

 public:
  explicit sha256_ostream(llvm::SHA256& sha256)
      : llvm::raw_ostream(/* unbuffered */ false), obj_(sha256) {
    // SetUnbuffered(); // copied from raw_svector_ostream
  }

  uint64_t bitcode_size() const { return pos_; }
  ~sha256_ostream() override { llvm::raw_ostream::flush(); }
};

}  // anonymous namespace

namespace amdgpu {

// Links ROCm-Device-Libs into the given module if the module needs it.
absl::Status LinkROCDLIfNecessary(llvm::Module* module,
                                  const std::string& gfx_version,
                                  const DebugOptions& debug_options,
                                  const std::string& rocdl_dir_path) {
  if (!CouldNeedDeviceBitcode(*module)) {
    return absl::OkStatus();
  }

  auto addControlVariable = [&](llvm::StringRef name, uint32_t value,
                                uint32_t bitwidth = 8) {
    if (module->getNamedGlobal(name)) {
      return;
    }
    llvm::IntegerType* type =
        llvm::IntegerType::getIntNTy(module->getContext(), bitwidth);
    llvm::GlobalVariable* control_variable = new llvm::GlobalVariable(
        *module, type, /*isConstant=*/true,
        llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
        llvm::ConstantInt::get(type, value), name, /*before=*/nullptr,
        /*threadLocalMode=*/llvm::GlobalValue::ThreadLocalMode::NotThreadLocal,
        /*addressSpace=*/4);
    control_variable->setVisibility(
        llvm::GlobalValue::VisibilityTypes::ProtectedVisibility);
    control_variable->setAlignment(llvm::MaybeAlign(bitwidth / 8));
    control_variable->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
  };

  addControlVariable("__oclc_finite_only_opt", false);
  // TODO(rocm): Maybe check ftz for this one
  addControlVariable("__oclc_daz_opt", false);
  addControlVariable("__oclc_correctly_rounded_sqrt32", true);
  addControlVariable("__oclc_unsafe_math_opt", false);

  auto [major, minor, stepping] = llvm::AMDGPU::getIsaVersion(gfx_version);

  CHECK(major != 0) << "Could not parse gfx_version.";

  // TODO(rocm): Not great, not terrible
  addControlVariable("__oclc_wavefrontsize64", major == 9);
  addControlVariable("__oclc_ISA_version",
                     1000 * major + 100 * stepping + minor, 32);
  addControlVariable("__oclc_ABI_version", kAMDGPUAbiVersion, 32);

  if (debug_options.xla_gpu_use_embeded_device_lib()) {
    llvm::Linker linker(*module);
    auto device_lib = llvm::getLazyBitcodeModule(
        {kAMDGPUDeviceLibData, "device_lib"}, module->getContext());
    if (!device_lib) {
      return absl::InternalError("Error loading embeded device lib.");
    }
    if (linker.linkInModule(
            std::move(*device_lib), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module& M, const llvm::StringSet<>& GVS) {
              internalizeModule(M, [&GVS](const llvm::GlobalValue& GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      return absl::InternalError("Error linking embeded device lib.");
    }
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(
      LinkWithBitcodeVector(module, GetROCDLPaths(rocdl_dir_path)));

  // Sanitize stray metadata from the bitcode files
  if (auto* opencl_version = module->getNamedMetadata("opencl.ocl.version")) {
    module->eraseNamedMetadata(opencl_version);
  }

  if (auto* ident = module->getNamedMetadata("llvm.ident")) {
    module->eraseNamedMetadata(ident);
  }

  return absl::OkStatus();
}

std::vector<std::string> GetAMDGPUBackendOptions(
    const DebugOptions& debug_options) {
  auto backend_llvm_opts = llvm_ir::ExtractXlaBackendExtraOptions(
      debug_options.xla_backend_extra_options());

  // Manually add LLVM debug options for register usage analysis
  // Note: The disassembly-based spilling detection is now the primary method.
  // These options are mainly useful for debugging the compiler itself.

  // Uncomment if you want to see LLVM compilation details:

  // Option 1: Enable LLVM statistics (aggregate stats, not per-kernel)
  // backend_llvm_opts.push_back("-stats");

  // Option 2: Print final machine code (very verbose)
  // backend_llvm_opts.push_back("-print-after-all");

  // Option 3: Print after register allocation (shows register assignments)
  // backend_llvm_opts.push_back("-print-after=regallocfast");
  // backend_llvm_opts.push_back("-print-after=regallocgreedy");

  // Option 4: Enable pass timing (shows compilation time breakdown)
  // backend_llvm_opts.push_back("-time-passes");

  // Log the final LLVM options
  if (!backend_llvm_opts.empty()) {
    LOG(INFO) << "AMDGPU backend LLVM options (" << backend_llvm_opts.size()
              << "):";
    for (const auto& opt : backend_llvm_opts) {
      LOG(INFO) << "  " << opt;
    }
  }

  return backend_llvm_opts;
}

absl::StatusOr<HsacoResult> CompileToHsaco(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& /*module_config_cache_key*/) {
  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
      tsl::profiler::TraceMeLevel::kInfo);

  auto llvm_opts = GetAMDGPUBackendOptions(debug_options);
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  auto comp_c = gpu_version.rocm_compute_capability();
  if (!comp_c) {
    return xla::Internal("Incompatible compute capability was specified.");
  }

  llvm::SHA256 sha256;
  sha256_ostream os(sha256);
  llvm::WriteBitcodeToFile(*module, os);
  os.flush();
  auto bitcode_size = os.bitcode_size();

  sha256.update(comp_c->gcn_arch_name());
  for (const auto& s : llvm_opts) sha256.update(s);
  // NOTE: adding module_config_cache_key to the hash, invalidates the
  // persistent file cache.
  // sha256.update(module_config_cache_key);

  // Add all relevant parameters to the hash to be on the safe side
  for (int32_t param :
       {static_cast<int32_t>(debug_options.xla_gpu_use_inprocess_lld()),
        static_cast<int32_t>(
            debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling()),
        static_cast<int32_t>(debug_options.xla_backend_optimization_level())}) {
    sha256.update(llvm::ArrayRef(reinterpret_cast<const uint8_t*>(&param),
                                 sizeof(param)));
  }
  HsacoCache::HashType binary_hash = sha256.final();

  HsacoResult compile_result;
  std::string hash_str;
  auto& cache = HsacoCache::i();
  if (cache.Find(binary_hash, bitcode_size, &hash_str, &compile_result)) {
    VLOG(1) << "HSACO cache hit for module '" << module->getModuleIdentifier()
            << "' (arch=" << comp_c->gcn_arch_name() << ", hash=" << hash_str
            << ", size=" << compile_result.hsaco.size() << " bytes)";
    // In case of file cache hit, ModuleStats need to be recomputed
    if (compile_result.module_stats.empty()) {
      compile_result.module_stats =
          ExtractRegisterSpillingFromHsaco(compile_result.hsaco)
              .ToModuleStats();
    }
    return compile_result;
  }

  VLOG(1) << "HSACO cache miss for module '" << module->getModuleIdentifier()
          << "' (arch=" << comp_c->gcn_arch_name() << ")";

  std::string hsaco_temp_path;
  auto compile_status = CompileToHsacoInternal(module, gpu_version,
                                               debug_options, &hsaco_temp_path);

  if (compile_status.ok()) {
    cache.Insert(binary_hash, bitcode_size, hsaco_temp_path,
                 cache.HsacoFilePath(hash_str), *compile_status);
  }
  if (!cache.KeepTempFiles()) {
    std::remove(hsaco_temp_path.c_str());
  }
  return compile_status;
}

}  // namespace amdgpu
}  // namespace gpu
}  // namespace xla
