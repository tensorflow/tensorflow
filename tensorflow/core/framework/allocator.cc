#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

Allocator::~Allocator() {}

class CPUAllocator : public Allocator {
 public:
  ~CPUAllocator() override {}

  string Name() override { return "cpu"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return port::aligned_malloc(num_bytes, alignment);
  }

  void DeallocateRaw(void* ptr) override { port::aligned_free(ptr); }
};

Allocator* cpu_allocator() {
  static CPUAllocator* cpu_alloc = new CPUAllocator;
  return cpu_alloc;
}

}  // namespace tensorflow
