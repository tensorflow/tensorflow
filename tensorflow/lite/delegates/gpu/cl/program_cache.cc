/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/program_cache.h"

#include <cstdint>
#include <string>
#include <utility>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/cl/compiled_program_cache_generated.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include <farmhash.h>

namespace tflite {
namespace gpu {
namespace cl {
namespace {

// Farmhash Fingerprint
inline uint64_t CombineFingerprints(uint64_t l, uint64_t h) {
  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (l ^ h) * kMul;
  a ^= (a >> 47);
  uint64_t b = (h ^ a) * kMul;
  b ^= (b >> 44);
  b *= kMul;
  b ^= (b >> 41);
  b *= kMul;
  return b;
}

uint64_t GetProgramFingerprint(const std::string& code,
                               const std::string& compiler_options) {
  const uint64_t code_fingerprint = ::util::Fingerprint64(code);
  const uint64_t options_fingerprint =
      ::util::Fingerprint64(compiler_options);
  return CombineFingerprints(code_fingerprint, options_fingerprint);
}

std::string GetDriverVersion(const CLDevice& device) {
  return device.GetPlatformVersion() + "_jet_version_0";
}

}  // namespace

ProgramCache::ProgramDescriptor::ProgramDescriptor(
    const std::string& code, const std::string& compiler_options)
    : fingerprint(GetProgramFingerprint(code, compiler_options)) {}

ProgramCache::ProgramDescriptor::ProgramDescriptor(uint64_t fingerprints)
    : fingerprint(fingerprints) {}

ProgramCache::ProgramCache(ProgramCache&& program_cache)
    : programs_(std::move(program_cache.programs_)) {}

ProgramCache& ProgramCache::operator=(ProgramCache&& program_cache) {
  if (this != &program_cache) {
    programs_ = std::move(program_cache.programs_);
  }
  return *this;
}

absl::Status ProgramCache::GetOrCreateCLKernel(
    const std::string& code, const std::string& function_name,
    const std::vector<CompilerOptions>& compiler_options,
    const CLContext& context, const CLDevice& device, CLKernel* result,
    uint64_t* kernel_fingerprint) {
  const std::string options =
      CompilerOptionsToString(device.GetInfo(), compiler_options);
  ProgramDescriptor desc(code, options);
  if (kernel_fingerprint) {
    *kernel_fingerprint = desc.fingerprint;
  }
  auto it = programs_.find(desc);
  if (it != programs_.end()) {
    return result->CreateFromProgram(it->second, function_name);
  }

  CLProgram program;
  RETURN_IF_ERROR(CreateCLProgram(code, options, context, device, &program));
  RETURN_IF_ERROR(result->CreateFromProgram(program, function_name));
  programs_.insert(std::make_pair(std::move(desc), std::move(program)));
  return absl::OkStatus();
}

absl::Status ProgramCache::GetOrCreateCLKernel(const std::string& code,
                                               const std::string& function_name,
                                               const CLContext& context,
                                               const CLDevice& device,
                                               CLKernel* result,
                                               uint64_t* kernel_fingerprint) {
  return GetOrCreateCLKernel(code, function_name, {}, context, device, result,
                             kernel_fingerprint);
}

absl::Status ProgramCache::GetKernel(uint64_t fingerprint,
                                     const std::string& function_name,
                                     CLKernel* result) const {
  ProgramDescriptor desc(fingerprint);
  auto it = programs_.find(desc);
  if (it == programs_.end()) {
    return absl::NotFoundError("No program with this fingerprint.");
  }
  return result->CreateFromProgram(it->second, function_name);
}

absl::Status ProgramCache::AddProgramBinary(const CLContext& context,
                                            const CLDevice& device,
                                            uint64_t fingerprint,
                                            absl::Span<const uint8_t> binary) {
  ProgramDescriptor desc(fingerprint);
  auto it = programs_.find(desc);
  if (it == programs_.end()) {
    CLProgram program;
    RETURN_IF_ERROR(
        CreateCLProgramFromBinary(context, device, binary, &program));
    programs_.insert(std::make_pair(std::move(desc), std::move(program)));
  }
  return absl::OkStatus();
}

absl::Status ProgramCache::GetProgramBinary(
    uint64_t fingerprint, std::vector<uint8_t>* program_binary) const {
  ProgramDescriptor desc(fingerprint);
  auto it = programs_.find(desc);
  if (it == programs_.end()) {
    return absl::NotFoundError("No program with this fingerprint.");
  }
  return it->second.GetBinary(program_binary);
}

absl::Status ProgramCache::AddSerializedCache(
    const CLContext& context, const CLDevice& device,
    absl::Span<const uint8_t> serialized_cache) {
  flatbuffers::Verifier verifier(serialized_cache.data(),
                                 serialized_cache.size());
  if (!data::VerifyCompiledCacheBuffer(verifier)) {
    return absl::InvalidArgumentError("Serialized model is corrupted.");
  }

  auto model = data::GetCompiledCache(serialized_cache.data());
  std::string platform_version(model->driver_version()->c_str(),
                               model->driver_version()->size());

  if (GetDriverVersion(device) != platform_version) {
    return absl::InvalidArgumentError(
        "OpenCL driver changed, cache invalid, should be regenerated");
  }

  for (auto serialized_program : *model->programs()) {
    auto binary_span = absl::MakeSpan(serialized_program->binary()->data(),
                                      serialized_program->binary()->size());
    RETURN_IF_ERROR(AddProgramBinary(
        context, device, serialized_program->fingerprint(), binary_span));
  }
  return absl::OkStatus();
}

absl::Status ProgramCache::GetSerializedCache(
    const CLDevice& device, std::vector<uint8_t>* serialized_cache) const {
  ::flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<data::Program>> serialized_programs;
  for (auto& program : programs_) {
    std::vector<uint8_t> binary;
    RETURN_IF_ERROR(program.second.GetBinary(&binary));
    auto binary_offset = builder.CreateVector(binary);
    data::ProgramBuilder program_builder(builder);
    program_builder.add_fingerprint(program.first.fingerprint);
    program_builder.add_binary(binary_offset);
    serialized_programs.push_back(program_builder.Finish());
  }
  auto driver_version = builder.CreateString(GetDriverVersion(device));
  auto programs_s = builder.CreateVector(serialized_programs);
  data::CompiledCacheBuilder cache_builder(builder);
  cache_builder.add_driver_version(driver_version);
  cache_builder.add_programs(programs_s);
  data::FinishCompiledCacheBuffer(builder, cache_builder.Finish());
  size_t next_element = serialized_cache->size();
  serialized_cache->resize(serialized_cache->size() + builder.GetSize());
  std::memcpy(&(*serialized_cache)[next_element], builder.GetBufferPointer(),
              builder.GetSize());
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
