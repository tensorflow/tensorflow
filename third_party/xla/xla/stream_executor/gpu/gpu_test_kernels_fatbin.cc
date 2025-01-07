/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_test_kernels_fatbin.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {

absl::StatusOr<std::vector<uint8_t>> GetGpuTestKernelsFatbin() {
  tsl::Env* env = tsl::Env::Default();
  std::string file_path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "stream_executor", "gpu",
                        "gpu_test_kernels.so");

  std::string file_contents;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, file_path, &file_contents));

  const auto buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(file_contents),
      /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  auto object_file =
      llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef());

  if (!object_file) {
    return absl::InternalError(llvm::toString(object_file.takeError()));
  }

  const auto executable_elf_object_file =
      llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(object_file.get().get());

  if (!executable_elf_object_file) {
    return absl::InternalError(
        "Generated executable binary is not a 64bit ELF file.");
  }

  for (const auto& section : executable_elf_object_file->sections()) {
    if (absl::StartsWith(section.getName().get().str(), ".nv_fatbin") ||
        absl::StartsWith(section.getName().get().str(), ".hip_fatbin")) {
      const std::string fatbin_contents = section.getContents().get().str();
      return std::vector<uint8_t>(fatbin_contents.begin(),
                                  fatbin_contents.end());
    }
  }

  return absl::InternalError("Fatbin section not found in generated ELF file.");
}
}  // namespace stream_executor::gpu
