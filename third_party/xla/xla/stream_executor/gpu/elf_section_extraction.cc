/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/elf_section_extraction.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

namespace stream_executor::gpu {

static absl::StatusOr<std::string> GetSectionContentsFromELFDump(
    const llvm::object::Binary& binary, absl::string_view section_name) {
  const llvm::object::ELF64LEObjectFile* executable_elf_object_file_ptr =
      llvm::dyn_cast<const llvm::object::ELF64LEObjectFile>(&binary);

  if (executable_elf_object_file_ptr == nullptr) {
    return absl::InternalError(
        "The binary is not a 64bit little endian ELF file - the only supported "
        "type.");
  }

  for (const auto& section : executable_elf_object_file_ptr->sections()) {
    if (section.getName().get().str() == section_name) {
      return section.getContents().get().str();
    }
  }

  return absl::NotFoundError(absl::StrFormat(
      "The section with the name %s not found in the binary.", section_name));
}

// Reads an archive file, iterates over all the files in the archive and
// searches for a section that is called `section_name`.
// Returns the contents of that section as a string in the first found ELF file.
// Returns NotFoundError if the section is not found in any of the files in the
// archive. Can also return other errors if the archive is malformed.
static absl::StatusOr<std::string> GetSectionContentsFromStaticLibraryDump(
    const llvm::object::Archive& archive, absl::string_view section_name) {
  llvm::Error archive_error = llvm::Error::success();
  for (const llvm::object::Archive::Child& child :
       archive.children(archive_error)) {
    if (archive_error) {
      return absl::InternalError(llvm::toString(std::move(archive_error)));
    }

    llvm::Expected<std::unique_ptr<llvm::object::Binary>> binary =
        child.getAsBinary();
    if (!binary) {
      continue;
    }

    absl::StatusOr<std::string> result =
        GetSectionContentsFromELFDump(**binary, section_name);
    if (absl::IsNotFound(result.status())) {
      continue;
    }
    return result;
  }

  return absl::InternalError(absl::StrFormat(
      "A section with the name %s was not found in the given archive.",
      section_name));
}

absl::StatusOr<std::string> GetSectionContentsFromLibraryDump(
    absl::string_view library_dump, absl::string_view section_name) {
  const auto buffer = llvm::MemoryBuffer::getMemBuffer(
      library_dump,
      /*BufferName=*/"", /*RequiresNullTerminator=*/false);

  llvm::Expected<std::unique_ptr<llvm::object::Binary>> binary =
      llvm::object::createBinary(buffer->getMemBufferRef());
  if (!binary) {
    return absl::InternalError(llvm::toString(binary.takeError()));
  }

  if (binary.get()->isArchive()) {
    return GetSectionContentsFromStaticLibraryDump(
        *llvm::cast<const llvm::object::Archive>(binary.get().get()),
        section_name);
  }
  if (binary.get()->isELF()) {
    return GetSectionContentsFromELFDump(**binary, section_name);
  }
  return absl::UnimplementedError(
      "Only ELF files and static library archives are supported.");
}
}  // namespace stream_executor::gpu
