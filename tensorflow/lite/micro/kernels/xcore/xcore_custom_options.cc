#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

#include "flatbuffers/flexbuffers.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

//*****************************
// PoolingParams
//*****************************
void parse_custom_options(const char *buffer, size_t length,
                          ::xcore::pooling::PoolingParams &pooling_params) {
  parse_custom_options(buffer, length, &pooling_params.stride_h,
                       &pooling_params.stride_w, &pooling_params.pool_h,
                       &pooling_params.pool_w,
                       nullptr,  // K_w
                       nullptr,  // pad
                       nullptr   // par_regions
  );
}

//*****************************
// Conv2DParams
//*****************************
void parse_custom_options(const char *buffer, size_t length,
                          ::xcore::conv::Conv2DParams &conv2d_params,
                          ::xcore::ParRegionArray *par_regions) {
  parse_custom_options(buffer, length, &conv2d_params.stride_h,
                       &conv2d_params.stride_w,
                       nullptr,  // pool_h
                       nullptr,  // pool_w
                       &conv2d_params.K_w, &conv2d_params.pad, par_regions);
}

//*****************************
// All Params
//*****************************
void parse_custom_options(const char *buffer, size_t length, int32_t *stride_h,
                          int32_t *stride_w, int32_t *pool_h, int32_t *pool_w,
                          int32_t *K_w, ::xcore::conv::Conv2DPadding *pad,
                          ::xcore::ParRegionArray *par_regions) {
  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);
  // std::cout << flexbuffers::GetRoot(buffer_t, length).ToString() <<
  // std::endl;
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

  auto keys = map.Keys();
  auto values = map.Values();
  for (int i = 0; i < map.size(); ++i) {
    const std::string &key = keys[i].AsString().str();

    if (key.compare("stride") == 0) {
      const auto &vec =
          values[i].AsVector();  // values represent [stride_h, stride_w]
      if (stride_h) *stride_h = vec[0].AsInt32();
      if (stride_w) *stride_w = vec[1].AsInt32();
    } else if (key.compare("stride_h") == 0) {
      if (stride_h) *stride_h = values[i].AsInt32();
    } else if (key.compare("stride_w") == 0) {
      if (stride_w) *stride_w = values[i].AsInt32();
    } else if (key.compare("Kw") == 0) {
      if (K_w) *K_w = values[i].AsInt32();
    } else if (key.compare("pool") == 0) {
      const auto &vec =
          values[i].AsVector();  // values represent [pool_h, pool_w]
      if (pool_h) *pool_h = vec[0].AsInt32();
      if (pool_w) *pool_w = vec[1].AsInt32();
    } else if (key.compare("pad") == 0) {
      if (pad) {
        const auto &vec =
            values[i].AsVector();  // values represent [top, left, zero_point]
        pad->top = vec[0].AsInt32();
        pad->left = vec[1].AsInt32();
        pad->zero_point = vec[2].AsInt32();
      }
    } else if (key.compare("par_plan") == 0) {
      if (par_regions) {
        const auto &jobs = values[i].AsVector();
        par_regions->clear();
        for (int i = 0; i < jobs.size(); i++) {
          auto region =
              jobs[i].AsVector();  // values represent [top, left, rows, cols]
          par_regions->append({region[0].AsInt32(), region[1].AsInt32(),
                               region[2].AsInt32(), region[3].AsInt32()});
        }
      }
    }
  }
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
