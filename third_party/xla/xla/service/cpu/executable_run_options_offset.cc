#include "executable_run_options_offset.h"
#include "xla/executable_run_options.h"

namespace xla::cpu {

// Friend-injection trick to get a pointer-to-private-member for the *real*
// xla::ExecutableRunOptions::batch_size_.
template <typename Tag, typename Tag::type Ptr>
struct Linker {
  friend constexpr typename Tag::type get_offset(Tag) { return Ptr; }
};

struct BatchSizeTag {
  using type = int64_t xla::ExecutableRunOptions::*;
  friend constexpr type get_offset(BatchSizeTag);
};

// Instantiate template to expose &ExecutableRunOptions::batch_size_.
template struct Linker<BatchSizeTag, &xla::ExecutableRunOptions::batch_size_>;

size_t ExecutableRunOptionsBatchSizeOffset() {
  auto ptr = get_offset(BatchSizeTag{});
  // Compute offset in bytes from null pointer.
  return reinterpret_cast<size_t>(
      &(reinterpret_cast<xla::ExecutableRunOptions*>(0)->*ptr));
}

}  // namespace xla::cpu
