#ifndef XCORE_PAR_H_
#define XCORE_PAR_H_

// extern "C" {
//     #include <xcore/parallel.h>
// }

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr uint32_t max_jobs = 5;

typedef struct {
    uint32_t top;
    uint32_t left;
    uint32_t rows;
    uint32_t cols;
} ParRegion;

typedef struct {
    uint32_t n_regions;
    ParRegion regions[max_jobs];
} ParPlan;

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_PAR_H_
