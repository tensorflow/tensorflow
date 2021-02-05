#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

CustomOptionParser::CustomOptionParser(const char *buffer, size_t buffer_length)
    : map_(flexbuffers::Map::EmptyMap()) {
  map_ = flexbuffers::GetRoot(reinterpret_cast<const uint8_t *>(buffer),
                              buffer_length)
             .AsMap();
}

flexbuffers::Reference CustomOptionParser::parseNamedCustomOption(
    const std::string &name) const {
  return map_[name];
};

//*****************************
// ExecutionPlan only
//*****************************
void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, ExecutionPlan *plan) {
  parse_custom_options(context, buffer, length, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr,
                       plan  // ExecutionPlan
  );
}

//*****************************
// PoolingParams
//*****************************
void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, PoolingParams &pooling_params,
                          ExecutionPlan *plan) {
  parse_custom_options(context, buffer, length, &pooling_params.stride_h,
                       &pooling_params.stride_w, &pooling_params.pool_h,
                       &pooling_params.pool_w,
                       nullptr,  // K_w
                       nullptr,  // pad
                       plan      // ExecutionPlan
  );
}

//*****************************
// Conv2DParams
//*****************************
void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, Conv2DParams &conv2d_params,
                          ExecutionPlan *plan) {
  parse_custom_options(context, buffer, length, &conv2d_params.stride_h,
                       &conv2d_params.stride_w,
                       nullptr,  // pool_h
                       nullptr,  // pool_w
                       &conv2d_params.K_w, &conv2d_params.pad, plan);
}

//*****************************
// All Params
//*****************************
void parse_custom_options(TfLiteContext *context, const char *buffer,
                          size_t length, int32_t *stride_h, int32_t *stride_w,
                          int32_t *pool_h, int32_t *pool_w, int32_t *K_w,
                          Conv2DPadding *pad, ExecutionPlan *plan) {
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
    } else if (key.compare("par") == 0) {
      if (plan) {
        const auto &plan_map = values[i].AsMap();
        auto plan_keys = plan_map.Keys();
        auto plan_values = plan_map.Values();
        for (int j = 0; j < plan_map.size(); ++j) {
          const std::string &plan_key = plan_keys[j].AsString().str();
          if (plan_key.compare("th") == 0) {
            plan->SetNumThreads(plan_values[j].AsInt32());
          } else if (plan_key.compare("cg") == 0) {
            const auto &changrps = plan_values[j].AsVector();
            plan->changrps.Init(context, changrps.size());
            for (int k = 0; k < changrps.size(); k++) {
              auto changrp =
                  changrps[k].AsVector();  // values represent [start, end]
              plan->changrps.Append(
                  {k, changrp[0].AsInt32(),
                   changrp[1].AsInt32() - changrp[0].AsInt32() + 1});
            }
          } else if (plan_key.compare("rc") == 0) {
            const auto &regions = plan_values[j].AsVector();
            plan->regions.Init(context, regions.size());
            for (int k = 0; k < regions.size(); k++) {
              auto region =
                  regions[k]
                      .AsVector();  // values represent [top, left, rows, cols]
              plan->regions.Append({region[0].AsInt32(), region[1].AsInt32(),
                                    region[2].AsInt32(), region[3].AsInt32()});
            }
          }
        }
      }
    } else if (key.compare("mem") == 0) {
      if (plan) {
        const auto &vec = values[i].AsVector();  // values represent [weights
                                                 // scratch, bias scratch]
        plan->SetWeightsScratchSize(vec[0].AsInt32());
        plan->SetBiasScratchSize(vec[1].AsInt32());
      }
    }
  }
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
