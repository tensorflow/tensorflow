// Implementation of the pointer-to-implementation wrapper for the data-parallel
// kernel abstraction. KernelBase just delegates to the internal
// platform-specific implementation instance.

#include "tensorflow/stream_executor/kernel.h"

#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/lib/demangle.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {

bool KernelMetadata::registers_per_thread(int *registers_per_thread) const {
  if (has_registers_per_thread_) {
    *registers_per_thread = registers_per_thread_;
    return true;
  }

  return false;
}

void KernelMetadata::set_registers_per_thread(int registers_per_thread) {
  registers_per_thread_ = registers_per_thread;
  has_registers_per_thread_ = true;
}

bool KernelMetadata::shared_memory_bytes(int *shared_memory_bytes) const {
  if (has_shared_memory_bytes_) {
    *shared_memory_bytes = shared_memory_bytes_;
    return true;
  }

  return false;
}

void KernelMetadata::set_shared_memory_bytes(int shared_memory_bytes) {
  shared_memory_bytes_ = shared_memory_bytes;
  has_shared_memory_bytes_ = true;
}

static internal::KernelInterface *KernelImplementationFromPlatformKind(
    PlatformKind platform_kind) {
  if (platform_kind == PlatformKind::kCuda) {
    return (*internal::MakeCUDAKernelImplementation())();
  } else if (platform_kind == PlatformKind::kOpenCL ||
             platform_kind == PlatformKind::kOpenCLAltera) {
    return (*internal::MakeOpenCLKernelImplementation())();
  } else {
    LOG(FATAL) << "cannot create kernel implementation for platform kind: "
               << PlatformKindString(platform_kind);
  }
}

KernelBase::KernelBase(StreamExecutor *parent)
    : implementation_(
          KernelImplementationFromPlatformKind(parent->platform_kind())),
      parent_(parent) {
  DCHECK(parent_ != nullptr);
}

KernelBase::KernelBase(StreamExecutor *parent,
                       internal::KernelInterface *implementation)
    : implementation_(implementation), parent_(parent) {}

KernelBase::~KernelBase() {}

unsigned KernelBase::Arity() const { return implementation_->Arity(); }

void KernelBase::SetPreferredCacheConfig(KernelCacheConfig config) {
  return implementation_->SetPreferredCacheConfig(config);
}

KernelCacheConfig KernelBase::GetPreferredCacheConfig() const {
  return implementation_->GetPreferredCacheConfig();
}

// Prefix stub functions emitted by the CUDA splitter.
static const char *kStubPrefix = "__device_stub_";

void KernelBase::set_name(port::StringPiece name) {
  name_ = name.ToString();
  port::StringPiece stubless_name = name;
  if (name.starts_with(kStubPrefix)) {
    stubless_name.remove_prefix(strlen(kStubPrefix));
  }
  demangled_name_ = port::Demangle(stubless_name.data());
}

}  // namespace gputools
}  // namespace perftools
