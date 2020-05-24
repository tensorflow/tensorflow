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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_PROGRAM_CACHE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_PROGRAM_CACHE_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_context.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_kernel.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_program.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

class ProgramCache {
 public:
  ProgramCache() = default;

  ProgramCache(ProgramCache&& program_cache);
  ProgramCache& operator=(ProgramCache&& program_cache);
  ProgramCache(const ProgramCache&) = delete;
  ProgramCache& operator=(const ProgramCache&) = delete;

  absl::Status GetOrCreateCLKernel(
      const std::string& code, const std::string& function_name,
      const std::vector<CompilerOptions>& compiler_options,
      const CLContext& context, const CLDevice& device, CLKernel* result);

  absl::Status GetOrCreateCLKernel(const std::string& code,
                                   const std::string& function_name,
                                   const CLContext& context,
                                   const CLDevice& device, CLKernel* result);

  absl::Status AddSerializedCache(const CLContext& context,
                                  const CLDevice& device,
                                  absl::Span<const uint8_t> serialized_cache);
  absl::Status GetSerializedCache(const CLDevice& device,
                                  std::vector<uint8_t>* serialized_cache) const;

 private:
  struct ProgramDescriptor {
    ProgramDescriptor() = default;
    ProgramDescriptor(const std::string& code_text, const std::string& options,
                      bool use_fingerprint);
    explicit ProgramDescriptor(uint64_t fingerprint);

    std::string code;
    std::string compiler_options;
    uint64_t fingerprint;
    bool use_fingerprint;
  };
  struct ProgramDescriptorHasher {
    std::size_t operator()(const ProgramDescriptor& k) const {
      if (k.use_fingerprint) {
        return std::hash<uint64_t>()(k.fingerprint);
      } else {
        return std::hash<std::string>()(k.code) +
               std::hash<std::string>()(k.compiler_options);
      }
    }
  };
  struct ProgramDescriptorEqual {
    bool operator()(const ProgramDescriptor& a,
                    const ProgramDescriptor& b) const {
      if (a.use_fingerprint && b.use_fingerprint) {
        return a.fingerprint == b.fingerprint;
      } else {
        return a.compiler_options == b.compiler_options && a.code == b.code;
      }
    }
  };

  // There is a low probability of a hash collision when cache is deserialized
  // because only fingerprints are serialized instead of full source code.
  bool use_fingerprints_ = false;
  std::unordered_map<ProgramDescriptor, CLProgram, ProgramDescriptorHasher,
                     ProgramDescriptorEqual>
      programs_;
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_PROGRAM_CACHE_H_
