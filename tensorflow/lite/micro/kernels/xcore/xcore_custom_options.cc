#include "flatbuffers/flexbuffers.h"

#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

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
                       nullptr,  // unpadded_shape
                       nullptr,  // pad
                       nullptr,  // par_plan
                       nullptr   // padding_mode
  );
}

//*****************************
// Conv2DParams
//*****************************
void parse_custom_options(const char *buffer, size_t length,
                          ::xcore::conv::Conv2DParams &conv2d_params,
                          ::xcore::conv::Conv2DUnpaddedShape *unpadded_shape,
                          ::xcore::ParRegionArray *par_regions,
                          padding_mode_t *padding_mode) {
  parse_custom_options(
      buffer, length, &conv2d_params.stride_h, &conv2d_params.stride_w,
      nullptr,  // pool_h
      nullptr,  // pool_w
      unpadded_shape, &conv2d_params.pad, par_regions, padding_mode);
}

//*****************************
// All Params
//*****************************
void parse_custom_options(const char *buffer, size_t length, int32_t *stride_h,
                          int32_t *stride_w, int32_t *pool_h, int32_t *pool_w,
                          ::xcore::conv::Conv2DUnpaddedShape *unpadded_shape,
                          ::xcore::conv::Conv2DPadding *pad,
                          ::xcore::ParRegionArray *par_regions,
                          padding_mode_t *padding_mode) {
  const uint8_t *buffer_t = reinterpret_cast<const uint8_t *>(buffer);
  // std::cout << flexbuffers::GetRoot(buffer_t, length).ToString() <<
  // std::endl;
  auto map = flexbuffers::GetRoot(buffer_t, length).AsMap();

  auto keys = map.Keys();
  auto values = map.Values();
  for (int i = 0; i < map.size(); ++i) {
    const std::string& key = keys[i].AsString().str();

    if (key.compare("stride") == 0) {
      auto vec = values[i].AsVector();  // values represent [stride_h, stride_w]
      if (stride_h) *stride_h = vec[0].AsInt32();
      if (stride_w) *stride_w = vec[1].AsInt32();
    } else if (key.compare("stride_h") == 0) {
      if (stride_h) *stride_h = values[i].AsInt32();
    } else if (key.compare("stride_w") == 0) {
      if (stride_w) *stride_w = values[i].AsInt32();
    } else if (key.compare("pool") == 0) {
      auto vec = values[i].AsVector();  // values represent [pool_h, pool_w]
      if (pool_h) *pool_h = vec[0].AsInt32();
      if (pool_w) *pool_w = vec[1].AsInt32();
    } else if (key.compare("unpadded_shape") == 0) {
      if (unpadded_shape) {
        auto vec =
            values[i].AsVector();  // values represent [C_out, K_h, K_w, C_in]
        unpadded_shape->C_out = vec[0].AsInt32();
        unpadded_shape->K_h = vec[1].AsInt32();
        unpadded_shape->K_w = vec[2].AsInt32();
        unpadded_shape->C_in = vec[3].AsInt32();
      }
    } else if (key.compare("pad") == 0) {
      if (pad) {
        auto vec =
            values[i].AsVector();  // values represent [top, left, zero_point]
        pad->top = vec[0].AsInt32();
        pad->left = vec[1].AsInt32();
        pad->zero_point = vec[2].AsInt32();
      }
    } else if (key.compare("par_plan") == 0) {
      if (par_regions) {
        auto jobs = values[i].AsVector();
        par_regions->clear();
        for (int i = 0; i < par_regions->size; i++) {
          auto region = jobs[i].AsVector();
          par_regions->append({region[0].AsInt32(), region[1].AsInt32(),
                              region[2].AsInt32(), region[3].AsInt32()});
        }
      }
    } else if (key.compare("padding") == 0) {
      if (padding_mode) {
        const std::string& padding_mode_str = values[i].AsString().str();
        if (padding_mode_str.compare("VALID") == 0)
          *padding_mode = PADDING_VALID;
        else
          *padding_mode = PADDING_SAME;
      }
    }
  }
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
