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

#ifndef XLA_PJRT_PJRT_PHASE_COMPILE_H_
#define XLA_PJRT_PJRT_PHASE_COMPILE_H_

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"

// This file defines the C++ convenience wrappers and internal functions for
// the PJRT Phase Compile C API extension. It provides higher-level interfaces
// for interacting with the phase compilation features, bridging between the
// low-level C API and XLA's C++ components.

namespace pjrt {

class PhaseCompileExtensionWrapper {
 public:
  static absl::StatusOr<std::unique_ptr<PhaseCompileExtensionWrapper>> Create(
      const PJRT_Api* api);

  // Runs the compilation phase with the given phases 'phases_to_run' on the
  // input programs 'partial_programs_in' and returns the output programs.
  // This function first performs plugin-agnostic validation on the inputs and
  // phase names. Next, it triggers plugin-specific phase validators for input
  // compatibility before invoking the appropriate phase compilers.
  absl::StatusOr<std::vector<xla::PjRtPartialProgramProto>> RunPhases(
      xla::CompileOptions options,
      const std::vector<xla::PjRtPartialProgramProto>& partial_programs_in,
      const xla::PjRtTopologyDescription& topology,
      const std::vector<std::string>& phases_to_run);

  // Returns the names of all the phases following the order of their
  // registration.
  absl::StatusOr<std::vector<std::string>> GetPhaseNames();

  // Returns the reference to the phase compiler.
  const PJRT_PhaseCompiler* GetPhaseCompiler() const { return phase_compiler_; }

  // Releases the phase compiler and destroys the phase compile extension.
  ~PhaseCompileExtensionWrapper();

 private:
  PhaseCompileExtensionWrapper(
      const PJRT_Api* api,
      const PJRT_PhaseCompile_Extension* phase_compile_extension,
      const PJRT_PhaseCompiler* phase_compiler)
      : api_(api),
        phase_compile_extension_(phase_compile_extension),
        phase_compiler_(phase_compiler) {}

  const PJRT_Api* api_ = nullptr;  // Not owned.
  const PJRT_PhaseCompile_Extension* phase_compile_extension_ =
      nullptr;  // Not owned.
  // This pointer is managed by the `PhaseCompileExtensionWrapper`. It is
  // acquired through `PJRT_PhaseCompile_Get_Compiler` and released using
  // `PJRT_PhaseCompile_Destroy_Compiler` in the destructor.
  const PJRT_PhaseCompiler* phase_compiler_ = nullptr;

  // Private nested struct for managing C-style buffers (arrays of char*).
  // These buffers are used to pass data across the C API boundary.
  // They can be created either locally at the caller site
  // (`PhaseCompileExtensionWrapper::RunPhases` or
  // `PhaseCompileExtensionWrapper::GetPhaseNames`, e.g., for input programs or
  // phase names to run) or remotely by the PJRT  plugin's
  // `PJRT_PhaseCompile_Run_Phase` or `PJRT_PhaseCompile_Get_PhaseNames`
  // functions, e.g., for output programs or returned phase names).
  //
  // This RAII wrapper manages the deallocation of C-style buffers. It uses
  // `is_locally_owned` to determine the deallocation strategy:
  // - If `is_locally_owned` is true, the buffers were allocated by the caller
  //   code, and they are deallocated using C++ `delete[]` when the wrapper
  //   goes out of scope.
  // - If `is_locally_owned` is false, the buffers were allocated by the PJRT
  //   plugin, and they are deallocated by calling the plugin-provided C API
  //   function `PJRT_PhaseCompile_C_Buffers_Destroy`.
  struct PhaseCompileCBuffersWrapper {
    PhaseCompileCBuffersWrapper(
        const char** char_buffers, const size_t* char_buffer_sizes,
        size_t num_char_buffers, bool is_locally_owned,
        const PJRT_PhaseCompile_Extension* phase_compile_extension)
        : char_buffers(char_buffers),
          char_buffer_sizes(char_buffer_sizes),
          num_char_buffers(num_char_buffers),
          is_locally_owned(is_locally_owned),
          phase_compile_extension(phase_compile_extension) {}

    ~PhaseCompileCBuffersWrapper() {
      if (is_locally_owned) {
        assert(char_buffers != nullptr);
        assert(char_buffer_sizes != nullptr);

        for (size_t i = 0; i < num_char_buffers; ++i) {
          delete[] char_buffers[i];
        }
        delete[] char_buffer_sizes;
        delete[] char_buffers;
      } else {
        PJRT_PhaseCompile_C_Buffers_Destroy_Args destroy_args;
        destroy_args.struct_size =
            PJRT_PhaseCompile_C_Buffers_Destroy_Args_STRUCT_SIZE;
        destroy_args.extension_start = nullptr;
        destroy_args.char_buffers = char_buffers;
        destroy_args.char_buffer_sizes = char_buffer_sizes;
        destroy_args.num_char_buffers = num_char_buffers;
        phase_compile_extension->phase_compile_c_buffers_destroy(&destroy_args);
      }
    }

    // Array of pointers to C-style character buffers to be deallocated.
    const char** char_buffers = nullptr;
    const size_t* char_buffer_sizes = nullptr;
    size_t num_char_buffers = 0;
    bool is_locally_owned = false;

    const PJRT_PhaseCompile_Extension* phase_compile_extension =
        nullptr;  // Not owned.

    PhaseCompileCBuffersWrapper(const PhaseCompileCBuffersWrapper&) = delete;
    PhaseCompileCBuffersWrapper& operator=(const PhaseCompileCBuffersWrapper&) =
        delete;
    PhaseCompileCBuffersWrapper(PhaseCompileCBuffersWrapper&&) = delete;
    PhaseCompileCBuffersWrapper& operator=(PhaseCompileCBuffersWrapper&&) =
        delete;
  };
};

}  // namespace pjrt

#endif  // XLA_PJRT_PJRT_PHASE_COMPILE_H_
