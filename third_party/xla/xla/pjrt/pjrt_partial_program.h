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

#ifndef XLA_PJRT_PJRT_PARTIAL_PROGRAM_H_
#define XLA_PJRT_PJRT_PARTIAL_PROGRAM_H_

#include <cstring>
#include <string>
#include <vector>

namespace xla {

class PjRtPartialProgram {
 public:
  PjRtPartialProgram() = default;
  PjRtPartialProgram(const PjRtPartialProgram& other) = default;
  PjRtPartialProgram& operator=(const PjRtPartialProgram& other) = default;
  PjRtPartialProgram(PjRtPartialProgram&& other) = default;
  PjRtPartialProgram& operator=(PjRtPartialProgram&& other) = default;
  ~PjRtPartialProgram() = default;

  std::string GetProgram() const {
    return std::string(program_, program_size_);
  }
  const char* GetProgramBuffer() const { return program_; }
  size_t GetProgramBufferSize() const { return program_size_; }
  size_t GetFormat() const { return format_; }
  const std::string& GetGeneratingPhase() const { return generating_phase_; }
  const std::vector<std::string>& GetNextPhases() const { return next_phases_; }
  const std::string& GetVersion() const { return version_; }

  void SetProgram(const std::string& program) {
    char* program_buffer = new char[program.size()];
    memcpy(program_buffer, program.data(), program.size());
    program_ = program_buffer;
    program_size_ = program.size();
  }
  void SetProgramBuffer(const char* program_buffer) {
    program_ = program_buffer;
  }
  void SetProgramBufferSize(size_t program_size) {
    program_size_ = program_size;
  }
  void SetFormat(size_t format) { format_ = format; }
  void SetGeneratingPhase(const std::string& phase) {
    generating_phase_ = phase;
  }
  void SetNextPhases(const std::vector<std::string>& phases) {
    next_phases_ = phases;
  }
  void SetVersion(const std::string& version) { version_ = version; }
  void Destroy() {
    delete[] program_;
    program_ = nullptr;
  }

 private:
  const char* program_;
  size_t program_size_;
  size_t format_;
  std::string generating_phase_;
  std::vector<std::string> next_phases_;
  std::string version_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_PARTIAL_PROGRAM_H_
