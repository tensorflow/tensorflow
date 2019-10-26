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

#include "flatbuffers/flatbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/cl/compiled_program_cache_generated.h"
#include "tensorflow/lite/delegates/gpu/cl/util.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include <farmhash.h>

namespace tflite {
namespace gpu {
namespace cl {

ProgramCache::ProgramDescriptor::ProgramDescriptor(const std::string& code_text,
                                                   const std::string& options,
                                                   bool use_fingerprints)
    : code(code_text),
      compiler_options(options),
      use_fingerprint(use_fingerprints) {
  const uint64_t code_fingerprint = ::util::Fingerprint64(code);
  const uint64_t options_fingerprint =
      ::util::Fingerprint64(compiler_options);
  fingerprint = code_fingerprint + options_fingerprint;
}

ProgramCache::ProgramDescriptor::ProgramDescriptor(uint64_t fingerprints)
    : fingerprint(fingerprints), use_fingerprint(true) {}

ProgramCache::ProgramCache(ProgramCache&& program_cache)
    : use_fingerprints_(program_cache.use_fingerprints_),
      programs_(std::move(program_cache.programs_)) {}

ProgramCache& ProgramCache::operator=(ProgramCache&& program_cache) {
  if (this != &program_cache) {
    use_fingerprints_ = program_cache.use_fingerprints_;
    programs_ = std::move(program_cache.programs_);
  }
  return *this;
}

Status ProgramCache::GetOrCreateCLKernel(
    const std::string& code, const std::string& function_name,
    const std::vector<CompilerOptions>& compiler_options,
    const CLContext& context, const CLDevice& device, CLKernel* result) {
  const std::string options = CompilerOptionsToString(device, compiler_options);
  ProgramDescriptor desc{code, options, use_fingerprints_};
  auto it = programs_.find(desc);
  if (it != programs_.end()) {
    RETURN_IF_ERROR(result->CreateFromProgram(it->second, function_name));
    return OkStatus();
  }

  CLProgram program;
  RETURN_IF_ERROR(CreateCLProgram(code, options, context, device, &program));
  RETURN_IF_ERROR(result->CreateFromProgram(program, function_name));
  programs_.insert(std::make_pair(std::move(desc), std::move(program)));
  return OkStatus();
}

Status ProgramCache::GetOrCreateCLKernel(const std::string& code,
                                         const std::string& function_name,
                                         const CLContext& context,
                                         const CLDevice& device,
                                         CLKernel* result) {
  return GetOrCreateCLKernel(code, function_name, {}, context, device, result);
}

Status ProgramCache::AddSerializedCache(
    const CLContext& context, const CLDevice& device,
    absl::Span<const uint8_t> serialized_cache) {
  flatbuffers::Verifier verifier(serialized_cache.data(),
                                 serialized_cache.size());
  if (!data::VerifyCompiledCacheBuffer(verifier)) {
    return InvalidArgumentError("Serialized model is corrupted.");
  }

  auto model = data::GetCompiledCache(serialized_cache.data());
  std::string platform_version(model->driver_version()->c_str(),
                               model->driver_version()->size());

  if (device.GetPlatformVersion() != platform_version) {
    return InvalidArgumentError(
        "OpenCL driver changed, cache invalid, should be regenerated");
  }

  use_fingerprints_ = true;

  for (auto serialized_program : *model->programs()) {
    ProgramDescriptor desc(serialized_program->fingerprint());
    CLProgram program;
    RETURN_IF_ERROR(CreateCLProgramFromBinary(
        context, device,
        absl::MakeSpan(serialized_program->binary()->data(),
                       serialized_program->binary()->size()),
        &program));
    auto it = programs_.find(desc);
    if (it == programs_.end()) {
      programs_.insert(std::make_pair(std::move(desc), std::move(program)));
    }
  }
  return OkStatus();
}

Status ProgramCache::GetSerializedCache(
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
  auto driver_version = builder.CreateString(device.GetPlatformVersion());
  auto programs_s = builder.CreateVector(serialized_programs);
  data::CompiledCacheBuilder cache_builder(builder);
  cache_builder.add_driver_version(driver_version);
  cache_builder.add_programs(programs_s);
  data::FinishCompiledCacheBuffer(builder, cache_builder.Finish());
  size_t next_element = serialized_cache->size();
  serialized_cache->resize(serialized_cache->size() + builder.GetSize());
  memcpy(&(*serialized_cache)[next_element], builder.GetBufferPointer(),
         builder.GetSize());
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
