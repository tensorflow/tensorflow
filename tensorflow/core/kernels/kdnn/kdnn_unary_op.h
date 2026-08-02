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

#ifndef TENSORFLOW_CORE_KERNELS_KDNN_KDNN_UNARY_OP_H_
#define TENSORFLOW_CORE_KERNELS_KDNN_KDNN_UNARY_OP_H_

// Templated kernel for KDNN-backed unary element-wise activations.
//
// This file mirrors the style of tensorflow/core/kernels/cwise_op*.h
// but dispatches through the KDNN C ABI (see third_party/KDNN/kdnn.h)
// instead of using Eigen's functor chain. KDNN is a single library for
// all activation ops on ARM/aarch64, so the template parameter is a
// `kdnn_activation_t` enum value.
//
// IMPORTANT: libkdnn.so is loaded at *runtime* via dlopen(), NOT linked
// at compile time. The TF binary can therefore be built with
// --define=enable_kdnn=true on any platform, and if libkdnn.so is not
// present at runtime the kernel falls back to a clear Unimplemented
// error (the Grappler remapper simply does not fire either, since
// IsKDNNEnabled() will be false in that case).
//
// Failure modes:
//   * IsKDNNEnabled() == false           -> kernel construction fails.
//   * dlopen() of libkdnn.so fails       -> SetStatus(Unimplemented).
//   * kdnn_apply_activation() returns
//       KDNN_ERR_UNSUPPORTED              -> SetStatus(Unimplemented).
//   * kdnn_apply_activation() returns
//       any other non-OK                  -> SetStatus(Internal) with the
//                                            kdnn_get_last_error() message.

#ifdef KERNEL_KDNN

#include <cstddef>
#include <cstdint>

#include <dlfcn.h>  // dlopen / dlsym / dlclose

#include "third_party/KDNN/kdnn.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"  // ReadBoolFromEnvVar
#include "tensorflow/core/util/port.h"      // IsKDNNEnabled

namespace tensorflow {
namespace functor {

// KDNN data-type tag -> kdnn_data_type_t enum. Only float / bfloat16 /
// half are mapped in the initial PR; complex / double / int fall
// through to a static_assert in the kernel and never reach KDNN.
template <typename T>
struct KdnnDataType;

template <>
struct KdnnDataType<float> {
  static constexpr kdnn_data_type_t value = KDNN_DT_FLOAT;
};
template <>
struct KdnnDataType<Eigen::half> {
  static constexpr kdnn_data_type_t value = KDNN_DT_HALF;
};
template <>
struct KdnnDataType<bfloat16> {
  static constexpr kdnn_data_type_t value = KDNN_DT_BFLOAT16;
};

}  // namespace functor

// =============================================================================
// KdnnDispatch — process-wide singleton holding the dlopen() handle and
// resolved function pointers for libkdnn.so.
// =============================================================================
//
// libkdnn.so is dlopen()ed lazily on the first kernel instantiation. The
// path to the library is, in order:
//   1. The env-var KDNN_LIB_PATH, if set.
//   2. The conventional location "libkdnn.so" (resolved by the dynamic
//      linker from LD_LIBRARY_PATH / RUNPATH / system loader paths).
//
// If either dlopen() or any required dlsym() fails, the singleton enters
// the "unavailable" state and every kernel instance returns
// Unimplemented from Compute(). This is the graceful-fallback path
// required by the design.
class KdnnDispatch {
 public:
  // Function-pointer signatures — must match the C ABI in kdnn.h.
  using CreateFn = kdnn_status_t (*)(kdnn_context_t**);
  using DestroyFn = void (*)(kdnn_context_t*);
  using GetLastErrorFn = const char* (*)(const kdnn_context_t*);
  using GetVersionFn = int (*)();
  using ActivationSupportedFn = int (*)(kdnn_activation_t, kdnn_data_type_t);
  using ApplyActivationFn = kdnn_status_t (*)(kdnn_context_t*,
                                               kdnn_activation_t,
                                               kdnn_data_type_t,
                                               kdnn_layout_t,
                                               void*, void*, size_t);

  // Returns the process-wide singleton. Initializes lazily on first call.
  static const KdnnDispatch& Get() {
    static const KdnnDispatch* instance = Create();
    return *instance;
  }

  // True if libkdnn.so was successfully loaded AND all required
  // symbols resolved. If false, kernels must return Unimplemented.
  bool IsAvailable() const { return handle_ != nullptr; }

  // Where the library was loaded from, or "" if not loaded. Useful for
  // log messages and tests.
  const std::string& Path() const { return path_; }

  // Function-pointer accessors. Safe to call only when IsAvailable().
  CreateFn kdnn_create() const { return kdnn_create_; }
  DestroyFn kdnn_destroy() const { return kdnn_destroy_; }
  GetLastErrorFn kdnn_get_last_error() const { return kdnn_get_last_error_; }
  GetVersionFn kdnn_get_version() const { return kdnn_get_version_; }
  ActivationSupportedFn kdnn_activation_supported() const {
    return kdnn_activation_supported_;
  }
  ApplyActivationFn kdnn_apply_activation() const {
    return kdnn_apply_activation_;
  }

 private:
  KdnnDispatch() = default;

  // Tries KDNN_LIB_PATH first; falls back to "libkdnn.so". Returns
  // nullptr on failure. `path_out` is set to the actually-tried path
  // (for diagnostics), or empty if the default name was used.
  static void* TryDlopen(std::string* path_out, std::string* error_out) {
    void* handle = nullptr;

    const char* env_path = std::getenv("KDNN_LIB_PATH");
    if (env_path != nullptr && env_path[0] != '\0') {
      *path_out = env_path;
      handle = dlopen(env_path, RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) {
        *error_out = dlerror();
        return nullptr;
      }
      return handle;
    }

    // Default: rely on the system loader. The first dlopen() of a
    // SONAME-only path is the standard pattern; LD_LIBRARY_PATH /
    // RUNPATH / system paths are honored.
    *path_out = "libkdnn.so";
    handle = dlopen("libkdnn.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      *error_out = dlerror();
      return nullptr;
    }
    return handle;
  }

  static const KdnnDispatch* Create() {
    auto* d = new KdnnDispatch();

    std::string path;
    std::string error;
    void* handle = TryDlopen(&path, &error);
    if (handle == nullptr) {
      // Graceful fallback. We deliberately do NOT call OP_REQUIRES or
      // signal an error here — the kernel will surface Unimplemented
      // when Compute() runs. Logging once is enough to alert the user.
      LOG(WARNING) << "KDNN library (" << path << ") could not be loaded: "
                   << (error.empty() ? "<no dlerror()>" : error)
                   << ". KDNN-backed ops will return Unimplemented.";
      // d->handle_ stays nullptr, IsAvailable() returns false.
      return d;
    }

    // Resolve every required symbol. If any fails, tear down and
    // treat the dispatch as unavailable — better to fail closed than
    // to call through a half-initialized table.
    auto resolve = [&](const char* name, void** out) {
      dlerror();  // clear
      *out = dlsym(handle, name);
      const char* err = dlerror();
      if (err != nullptr) {
        LOG(WARNING) << "KDNN library (" << path
                     << ") is missing required symbol '" << name
                     << "': " << err
                     << ". KDNN-backed ops will return Unimplemented.";
        dlclose(handle);
        handle = nullptr;
      }
    };

    void* p_create = nullptr;
    void* p_destroy = nullptr;
    void* p_last_err = nullptr;
    void* p_version = nullptr;
    void* p_supported = nullptr;
    void* p_apply = nullptr;

    resolve("kdnn_create", &p_create);
    if (handle == nullptr) return d;
    resolve("kdnn_destroy", &p_destroy);
    if (handle == nullptr) return d;
    resolve("kdnn_get_last_error", &p_last_err);
    if (handle == nullptr) return d;
    resolve("kdnn_get_version", &p_version);
    if (handle == nullptr) return d;
    resolve("kdnn_activation_supported", &p_supported);
    if (handle == nullptr) return d;
    resolve("kdnn_apply_activation", &p_apply);
    if (handle == nullptr) return d;

    d->handle_ = handle;
    d->path_ = path;
    d->kdnn_create_ = reinterpret_cast<CreateFn>(p_create);
    d->kdnn_destroy_ = reinterpret_cast<DestroyFn>(p_destroy);
    d->kdnn_get_last_error_ = reinterpret_cast<GetLastErrorFn>(p_last_err);
    d->kdnn_get_version_ = reinterpret_cast<GetVersionFn>(p_version);
    d->kdnn_activation_supported_ =
        reinterpret_cast<ActivationSupportedFn>(p_supported);
    d->kdnn_apply_activation_ = reinterpret_cast<ApplyActivationFn>(p_apply);

    LOG(INFO) << "KDNN library loaded from " << path
              << " (version 0x" << std::hex << d->kdnn_get_version_()
              << std::dec << ").";
    return d;
  }

  // Not copyable, not movable. We hold raw function pointers that must
  // remain valid for the lifetime of the process.
  KdnnDispatch(const KdnnDispatch&) = delete;
  KdnnDispatch& operator=(const KdnnDispatch&) = delete;

  void* handle_ = nullptr;
  std::string path_;

  CreateFn kdnn_create_ = nullptr;
  DestroyFn kdnn_destroy_ = nullptr;
  GetLastErrorFn kdnn_get_last_error_ = nullptr;
  GetVersionFn kdnn_get_version_ = nullptr;
  ActivationSupportedFn kdnn_activation_supported_ = nullptr;
  ApplyActivationFn kdnn_apply_activation_ = nullptr;
};

// =============================================================================
// KdnnUnaryOp<Activation, T> — concrete kernel class.
// =============================================================================
//
// `Activation` is a kdnn_activation_t enum value (e.g. KDNN_ACT_SIGMOID).
// `T` is the input/output data type (float / Eigen::half / bfloat16).
//
// All kernel state (the kdnn_context_t) is held once per kernel
// instance via the base class. We do NOT make it thread-local; the KDNN
// library is expected to be thread-safe per-context. If KDNN's thread
// safety changes in a future version, this class can switch to a
// per-thread context trivially.
template <kdnn_activation_t Activation, typename T>
class KdnnUnaryOp : public OpKernel {
 public:
  explicit KdnnUnaryOp(OpKernelConstruction* context) : OpKernel(context) {
    // Defensive: do not register this kernel unless KDNN is actually
    // enabled at runtime. Build-time gating is done via #ifdef
    // KERNEL_KDNN; runtime gating here is a belt-and-suspenders
    // measure. The kernel is only constructed if the device is CPU
    // (see REGISTER_KERNEL_BUILDER below).
    OP_REQUIRES(context, IsKDNNEnabled(),
                errors::FailedPrecondition(
                    "KDNN is not enabled at runtime (TF_ENABLE_KDNN_OPTS=0 "
                    "or build was done without --define=enable_kdnn=true)"));

    const KdnnDispatch& d = KdnnDispatch::Get();
    if (!d.IsAvailable()) {
      // libkdnn.so is not present at runtime. Don't construct a context;
      // Compute() will return Unimplemented.
      return;
    }

    kdnn_context_t* raw_ctx = nullptr;
    kdnn_status_t s = d.kdnn_create()(&raw_ctx);
    OP_REQUIRES(
        context, s == KDNN_OK && raw_ctx != nullptr,
        errors::Internal(absl::StrCat(
            "kdnn_create() failed: ",
            raw_ctx ? d.kdnn_get_last_error()(raw_ctx) : "<null context>",
            " (", static_cast<int>(s), ")")));
    ctx_ = raw_ctx;
  }

  ~KdnnUnaryOp() override {
    if (ctx_ != nullptr) {
      const KdnnDispatch& d = KdnnDispatch::Get();
      if (d.IsAvailable()) d.kdnn_destroy()(ctx_);
      ctx_ = nullptr;
    }
  }

  void Compute(OpKernelContext* context) override {
    const KdnnDispatch& d = KdnnDispatch::Get();
    if (!d.IsAvailable()) {
      // Graceful fallback: libkdnn.so was not present at process start.
      // Surface this as Unimplemented so the graph runtime can decide
      // whether to fall back to a different op. We deliberately do not
      // fail the whole op — eager-mode users see a clear message and
      // can opt out via TF_ENABLE_KDNN_OPTS=0 to silence it.
      context->SetStatus(errors::Unimplemented(absl::StrCat(
          "KDNN library (libkdnn.so) is not available at runtime. Tried path: ",
          d.Path().empty() ? "<default>" : d.Path(),
          ". Install KAIL BoostKit's libkdnn and either place it on the "
          "dynamic linker's search path or set the KDNN_LIB_PATH env var. "
          "Set TF_ENABLE_KDNN_OPTS=0 to disable this kernel.")));
      return;
    }

    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    const size_t n = static_cast<size_t>(input.shape().num_elements());
    if (n == 0) return;  // empty tensor — nothing to do.

    // Allow in-place: x == y is valid in KDNN.
    void* x = const_cast<void*>(static_cast<const void*>(input.flat<T>().data()));
    void* y = static_cast<void*>(output->flat<T>().data());

    kdnn_status_t s = d.kdnn_apply_activation()(
        ctx_, Activation, functor::KdnnDataType<T>::value,
        KDNN_LAYOUT_ROW_MAJOR, x, y, n);

    if (s == KDNN_ERR_UNSUPPORTED) {
      // The remapper should not have picked us. Fail with an actionable
      // error so the user can see the misconfiguration.
      context->SetStatus(errors::Unimplemented(absl::StrCat(
          "KDNN does not support activation ", static_cast<int>(Activation),
          " for dtype ", DataTypeString(DataTypeToEnum<T>::value),
          " on this CPU. The remapper should not have substituted this op.")));
      return;
    }

    OP_REQUIRES(
        context, s == KDNN_OK,
        errors::Internal(absl::StrCat(
            "kdnn_apply_activation() failed: ", d.kdnn_get_last_error()(ctx_),
            " (", static_cast<int>(s), ")")));
  }

 private:
  kdnn_context_t* ctx_ = nullptr;
};

}  // namespace tensorflow

#endif  // KERNEL_KDNN

#endif  // TENSORFLOW_CORE_KERNELS_KDNN_KDNN_UNARY_OP_H_