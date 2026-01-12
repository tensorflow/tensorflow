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
#include <memory>
#include <optional>
#include <string>
#include <system_error>  // NOLINT
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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

struct HsacoCacheEntry {
  uint64_t hash;
  std::string ir;
  std::string gfx;
  std::vector<uint8_t> hsaco;
};

struct HsacoCache {
 protected:
  std::vector<HsacoCacheEntry> cache ABSL_GUARDED_BY(mutex);
  absl::Mutex mutex;
  int request_count ABSL_GUARDED_BY(mutex) = 0;
  int hit_count ABSL_GUARDED_BY(mutex) = 0;

 public:
  static bool Find(const std::string& ir, uint64_t& hash,
                   const std::string& gfx, std::vector<uint8_t>& hsaco);
  static void Add(const std::string& ir, uint64_t hash, const std::string& gfx,
                  const std::vector<uint8_t>& hsaco);
};

static HsacoCache g_hsacoCache;  // NOLINT: static/global vars forbidden

// Structure to hold register spilling and stack information from HSACO metadata
struct RegisterSpillInfo {
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

// Parse NT_AMDGPU_METADATA note contents and extract register spill counts.
// The metadata is in MessagePack format containing kernel information.
RegisterSpillInfo ParseAMDGPUMetadataForSpills(llvm::StringRef metadata) {
  RegisterSpillInfo spill_info;

  // Parse the MsgPack metadata
  llvm::msgpack::Document doc;
  if (!doc.readFromBlob(metadata, /*Multi=*/false)) {
    VLOG(2) << "Could not parse MsgPack metadata from NT_AMDGPU_METADATA note";
    return spill_info;
  }

  llvm::msgpack::DocNode root = doc.getRoot();
  if (!root.isMap()) {
    VLOG(2) << "AMDGPU metadata root is not a map (unexpected format)";
    return spill_info;
  }

  // Look for "amdhsa.kernels" array
  llvm::msgpack::MapDocNode root_map = root.getMap();
  auto kernels_it = root_map.find("amdhsa.kernels");

  if (kernels_it == root_map.end() || !kernels_it->second.isArray()) {
    VLOG(2) << "NT_AMDGPU_METADATA found but missing 'amdhsa.kernels' array";
    return spill_info;
  }

  llvm::msgpack::ArrayDocNode kernels_array = kernels_it->second.getArray();

  // Iterate through each kernel
  for (auto& kernel_node : kernels_array) {
    uint64_t kernel_sgpr_spill = 0;
    uint64_t kernel_vgpr_spill = 0;
    uint64_t kernel_sgpr_count = 0;
    uint64_t kernel_vgpr_count = 0;
    uint64_t kernel_private_size = 0;
    bool kernel_uses_dynamic = false;

    if (!kernel_node.isMap()) continue;

    llvm::msgpack::MapDocNode kernel_map = kernel_node.getMap();

    // Look for ".sgpr_spill_count"
    auto sgpr_it = kernel_map.find(".sgpr_spill_count");
    if (sgpr_it != kernel_map.end() &&
        sgpr_it->second.getKind() == llvm::msgpack::Type::UInt) {
      kernel_sgpr_spill = sgpr_it->second.getUInt();
      spill_info.sgpr_spill_count =
          std::max(spill_info.sgpr_spill_count, kernel_sgpr_spill);
    }

    // Look for ".vgpr_spill_count"
    auto vgpr_it = kernel_map.find(".vgpr_spill_count");
    if (vgpr_it != kernel_map.end() &&
        vgpr_it->second.getKind() == llvm::msgpack::Type::UInt) {
      kernel_vgpr_spill = vgpr_it->second.getUInt();
      spill_info.vgpr_spill_count =
          std::max(spill_info.vgpr_spill_count, kernel_vgpr_spill);
    }

    // Look for ".private_segment_fixed_size"
    auto priv_it = kernel_map.find(".private_segment_fixed_size");
    if (priv_it != kernel_map.end() &&
        priv_it->second.getKind() == llvm::msgpack::Type::UInt) {
      kernel_private_size = priv_it->second.getUInt();
      spill_info.private_segment_size =
          std::max(spill_info.private_segment_size, kernel_private_size);
    }

    // Look for ".uses_dynamic_stack"
    auto dyn_it = kernel_map.find(".uses_dynamic_stack");
    if (dyn_it != kernel_map.end() &&
        dyn_it->second.getKind() == llvm::msgpack::Type::Boolean) {
      kernel_uses_dynamic = dyn_it->second.getBool();
      spill_info.uses_dynamic_stack =
          spill_info.uses_dynamic_stack || kernel_uses_dynamic;
    }

    // Helper to get kernel name for logging (only when needed)
    auto get_kernel_name = [&kernel_map]() -> std::string {
      auto name_it = kernel_map.find(".name");
      if (name_it != kernel_map.end() &&
          name_it->second.getKind() == llvm::msgpack::Type::String) {
        return name_it->second.getString().str();
      }
      return "unknown";
    };

    // Log per-kernel spill information with register usage
    if (kernel_sgpr_spill > 0 || kernel_vgpr_spill > 0) {
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

      VLOG(2) << "Kernel '" << get_kernel_name() << "' has register spilling: "
              << "SGPR=" << kernel_sgpr_spill << ", VGPR=" << kernel_vgpr_spill
              << ". Register count: SGPR=" << kernel_sgpr_count
              << ", VGPR=" << kernel_vgpr_count;
    }

    // Log per-kernel stack usage
    if (kernel_private_size > 0 || kernel_uses_dynamic) {
      VLOG(2) << "Kernel '" << get_kernel_name() << "' stack usage: "
              << "private=" << kernel_private_size
              << ", dynamic=" << (kernel_uses_dynamic ? "true" : "false");
    }
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
  RegisterSpillInfo spill_info;

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
    return spill_info;
  }

  llvm::object::ObjectFile* obj = obj_or_err->get();

  // Cast to ELF64LE object file (AMDGPU uses 64-bit little-endian ELF)
  auto* elf_obj = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(obj);
  if (!elf_obj) {
    VLOG(2) << "HSACO is not a 64-bit little-endian ELF file";
    return spill_info;
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
  return spill_info;
}

bool HsacoCache::Find(const std::string& ir, uint64_t& hash,
                      const std::string& gfx, std::vector<uint8_t>& hsaco) {
  absl::MutexLock lock(g_hsacoCache.mutex);
  hash = std::hash<std::string>{}(ir);
  bool hit = false;
  for (auto& x : g_hsacoCache.cache) {
    if (x.hash != hash) {
      continue;
    }
    if (x.gfx != gfx) {
      continue;
    }
    if (x.ir != ir) {
      continue;
    }
    hsaco = x.hsaco;
    hit = true;
    break;
  }
  g_hsacoCache.request_count++;
  if (hit) {
    g_hsacoCache.hit_count++;
  }
  if (!(g_hsacoCache.request_count % 50)) {
    VLOG(1) << "HSACO cache: " << g_hsacoCache.request_count << " requests, "
            << g_hsacoCache.hit_count << " hits";
  }
  return hit;
}

void HsacoCache::Add(const std::string& ir, uint64_t hash,
                     const std::string& gfx,
                     const std::vector<uint8_t>& hsaco) {
  absl::MutexLock lock(g_hsacoCache.mutex);
  g_hsacoCache.cache.resize(g_hsacoCache.cache.size() + 1);
  g_hsacoCache.cache.back().ir = ir;
  g_hsacoCache.cache.back().hash = hash;
  g_hsacoCache.cache.back().gfx = gfx;
  g_hsacoCache.cache.back().hsaco = hsaco;
}

// Emits the given module to HSA Code Object. target_machine is an initialized
// TargetMachine for the AMDGPU target.
absl::StatusOr<std::string> EmitModuleToHsaco(
    llvm::Module* module, llvm::TargetMachine* target_machine,
    const DebugOptions& debug_options, bool keep_tempfiles) {
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
  std::string ir_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".ll");
  std::string ir_path = tsl::io::JoinPath(tempdir_name, ir_filename);

  std::string ir_opt_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + "_opt.ll");
  std::string ir_opt_path = tsl::io::JoinPath(tempdir_name, ir_opt_filename);

  std::string isabin_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".o");
  std::string isabin_path = tsl::io::JoinPath(tempdir_name, isabin_filename);

  std::string hsaco_filename =
      absl::StrCat(module->getModuleIdentifier(), random_number + ".hsaco");
  std::string hsaco_path = tsl::io::JoinPath(tempdir_name, hsaco_filename);

  std::error_code ec;

  // Dump LLVM IR.
  std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
      new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::OF_None));
  module->print(*ir_fs, nullptr);
  ir_fs->flush();

  // Emit GCN ISA binary.
  llvm::legacy::PassManager pm;
  pm.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(module->getTargetTriple())));
  llvm::SmallVector<char, 0> stream;
  llvm::raw_svector_ostream pstream(stream);
  std::unique_ptr<llvm::raw_fd_ostream> isabin_fs(
      new llvm::raw_fd_ostream(isabin_path, ec, llvm::sys::fs::OF_Text));
  module->setDataLayout(target_machine->createDataLayout());
  target_machine->addPassesToEmitFile(pm, *isabin_fs, nullptr,
                                      llvm::CodeGenFileType::ObjectFile);
  pm.run(*module);
  isabin_fs->flush();

  if (keep_tempfiles) {
    std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
        new llvm::raw_fd_ostream(ir_opt_path, ec, llvm::sys::fs::OF_None));
    module->print(*ir_fs, nullptr);
    ir_fs->flush();
  }

  if (debug_options.xla_gpu_use_inprocess_lld()) {
#ifdef HAS_SUPPORT_FOR_LLD_AS_A_LIBRARY
    static absl::Mutex lld_mu(absl::kConstInit);

    std::array<const char*, 7> args{
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
#endif
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
    std::vector<llvm::StringRef> lld_args{
        llvm_ir::AsStringRef("ld.lld"),
        llvm_ir::AsStringRef("-flavor"),
        llvm_ir::AsStringRef("gnu"),
        llvm_ir::AsStringRef("-shared"),
        llvm_ir::AsStringRef("--no-undefined"),
        llvm_ir::AsStringRef(isabin_path),
        llvm_ir::AsStringRef("-o"),
        llvm_ir::AsStringRef(hsaco_path),
    };

    std::string error_message;
    int lld_result =
        llvm::sys::ExecuteAndWait(*lld_program, llvm_ir::AsArrayRef(lld_args),
                                  std::nullopt, {}, 0, 0, &error_message);
    if (lld_result) {
      return xla::Internal("ld.lld execute fail: %s, error code %d",
                           error_message, lld_result);
    }
  }

  // Read HSACO file into memory (used for both metadata extraction and return)
  std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
  if (!hsaco_file) {
    return xla::Internal("Failed to open HSACO file: %s", hsaco_path);
  }
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();
  std::vector<uint8_t> hsaco(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char*>(hsaco.data()), hsaco_file_size);
  hsaco_file.close();

  // Check for register spilling using HSACO metadata
  VLOG(2) << "Checking for register spilling in: "
          << module->getModuleIdentifier();

  RegisterSpillInfo spill_info = ExtractRegisterSpillingFromHsaco(hsaco);

  if (spill_info.HasSpilling()) {
    // We can have SGPR spills without stack being used. They are saved to
    // VGPRs. In that case, we don't want to discard such kernel, so just
    // report such cases.
    VLOG(1) << "Register spilling (SGPR: " << spill_info.sgpr_spill_count
            << ", VGPR: " << spill_info.vgpr_spill_count << ") detected in "
            << module->getModuleIdentifier();
  } else {
    VLOG(2) << "No register spilling detected in "
            << module->getModuleIdentifier();
  }

  if (spill_info.HasStackUsage()) {
    VLOG(1) << "Stack usage (private: " << spill_info.private_segment_size
            << ", dynamic: "
            << (spill_info.uses_dynamic_stack ? "true" : "false")
            << ") detected in " << module->getModuleIdentifier();

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

  // Clean up temp files
  if (!keep_tempfiles) {
    remove(ir_path.c_str());
    remove(isabin_path.c_str());
  }
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
  std::string feature_str;

  std::string gfx = gcn_arch_name;
  // For ROCm versions 4.0 and greater, we need to specify the correct
  // feature str, based on the underlying GPU HW to get max performance.
  std::vector<std::string> tokens = absl::StrSplit(gcn_arch_name, ':');
  std::vector<std::string> mapped_tokens;
  if (!tokens.empty()) {
    gfx = tokens[0];
  }
  for (auto it = tokens.begin(); it != tokens.end(); it++) {
    // Skip the first token, that is the gfxNNN str
    // The rest of the tokens are the feature/targetid strings
    if (it != tokens.begin()) {
      std::string token(*it);
      std::string mapped_token = MapGCNArchNameTokenToFeatureStr(token, gfx);
      mapped_tokens.push_back(mapped_token);
    }
  }
  feature_str = absl::StrJoin(mapped_tokens, ",");

  return std::make_pair(gfx, feature_str);
}

std::unique_ptr<llvm::TargetMachine> AMDGPUGetTargetMachine(
    llvm::Triple target_triple, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options) {
  auto compute_capability = gpu_version.rocm_compute_capability();

  std::string gcn_arch_name = compute_capability->gcn_arch_name();
  auto arch = GetFeatureStrFromGCNArchName(gcn_arch_name);
  return GetTargetMachine(std::move(target_triple), arch.first, debug_options,
                          arch.second);
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

}  // namespace

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
  std::vector<std::string> backend_llvm_opts;

  // Extra backend options must go after regular backend options in order to be
  // able for the later to override the former.
  auto backend_extra_llvm_opts = llvm_ir::ExtractXlaBackendExtraOptions(
      debug_options.xla_backend_extra_options());
  backend_llvm_opts.insert(backend_llvm_opts.end(),
                           backend_extra_llvm_opts.cbegin(),
                           backend_extra_llvm_opts.cend());

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

absl::StatusOr<std::vector<uint8_t>> CompileToHsaco(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options,
    const std::string& module_config_cache_key) {
  auto llvm_opts = GetAMDGPUBackendOptions(debug_options);

  VLOG(2) << "CompileToHsaco called for module: "
          << module->getModuleIdentifier();

  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_opts);

  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << *module;
  // Delete the first two lines, since they usually vary even when the rest of
  // the code is the same (but verify that they are what we expect).
  if (str.size() >= 13 && str.substr(0, 13) == "; ModuleID = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) {
      str = str.substr(pos + 1);
    }
  }
  if (str.size() >= 18 && str.substr(0, 18) == "source_filename = ") {
    auto pos = str.find('\n');
    if (pos != std::string::npos) {
      str = str.substr(pos + 1);
    }
  }
  str += module_config_cache_key;

  tsl::profiler::TraceMe activity(
      [&] { return absl::StrCat("Compiling IR", module->getName().str()); },
      tsl::profiler::TraceMeLevel::kInfo);
  XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

  auto compute_capability = gpu_version.rocm_compute_capability();
  if (!compute_capability) {
    return xla::Internal("Incompatible compute capability was specified.");
  }

  std::string gcn_arch_name = compute_capability->gcn_arch_name();

  uint64_t hash;
  std::vector<uint8_t> hsaco;
  if (HsacoCache::Find(str, hash, gcn_arch_name, hsaco)) {
    VLOG(1) << "HSACO cache hit";
    return hsaco;
  }
  VLOG(1) << "HSACO cache miss";
  bool dump_lls = false;
  if (dump_lls) {
    static int hsaco_count = 0;
    std::string name = "/tmp/" + std::to_string(hsaco_count) + ".ll";
    hsaco_count++;
    std::ofstream ofs(name);
    ofs << str;
    ofs.close();
  }

  bool keep_tempfiles = false;
  TF_CHECK_OK(tsl::ReadBoolFromEnvVar("TF_ROCM_KEEP_XLA_TEMPFILES",
                                      /*default_val=*/false, &keep_tempfiles));
  TF_ASSIGN_OR_RETURN(auto hsaco_output_path,
                      CompileToHsacoAndReturnFilePath(
                          module, gpu_version, debug_options, keep_tempfiles));

  // Read HSACO.
  std::ifstream hsaco_file(hsaco_output_path, std::ios::binary | std::ios::ate);
  std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();
  hsaco.resize(hsaco_file_size);
  hsaco_file.seekg(0, std::ios::beg);
  hsaco_file.read(reinterpret_cast<char*>(hsaco.data()), hsaco_file_size);
  hsaco_file.close();
  if (!keep_tempfiles) {
    remove(hsaco_output_path.c_str());
  }
  HsacoCache::Add(str, hash, gcn_arch_name, hsaco);

  return hsaco;
}

absl::StatusOr<std::string> CompileToHsacoAndReturnFilePath(
    llvm::Module* module, se::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options, bool keep_tempfiles) {
  static absl::once_flag backend_init_flag;
  // TODO(rocm) Ideally this would be refreshed if xla_gpu_cuda_data_dir
  // changes.
  static std::string rocdl_dir_path;  // NOLINT: static/global vars forbidden
  absl::call_once(backend_init_flag, AMDGPUBackendInit, debug_options,
                  rocdl_dir_path);

  llvm::Triple default_target_triple("amdgcn--amdhsa-amdgiz");
  // Construct LLVM TargetMachine for AMDGPU.
  std::unique_ptr<llvm::TargetMachine> target_machine =
      AMDGPUGetTargetMachine(default_target_triple, gpu_version, debug_options);

  // Link with ROCm-Device-Libs, and optimize the LLVM module.
  TF_RETURN_IF_ERROR(gpu::LinkAndOptimizeModule(
      module, gpu_version, debug_options, rocdl_dir_path,
      AMDGPUTargetModuleLinker, default_target_triple, target_machine.get(),
      kAMDGPUInlineThreshold));

  // Lower optimized LLVM module to HSA code object.
  TF_ASSIGN_OR_RETURN(auto hsaco_path,
                      EmitModuleToHsaco(module, target_machine.get(),
                                        debug_options, keep_tempfiles));
  return hsaco_path;
}

}  // namespace amdgpu
}  // namespace gpu
}  // namespace xla
