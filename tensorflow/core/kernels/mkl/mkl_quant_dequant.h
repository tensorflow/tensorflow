/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#ifdef DNNL_AARCH64_USE_ACL
#include "tensorflow/core/platform/mutex.h"
#endif

using dnnl::primitive_attr;
using dnnl::prop_kind;
using dnnl::reorder;
using dnnl::stream;

namespace {
enum {
  QUANTIZE_MODE_MIN_COMBINED,
  QUANTIZE_MODE_MIN_FIRST,
  QUANTIZE_MODE_SCALED,
};
enum {
  // Round half away from zero: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5 if y > 0
  // round(y) = y - 0.5 if y < 0
  // E.g., -5.5 gets rounded to -6, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_AWAY_FROM_ZERO,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};
}  // namespace

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

struct MklReorderWithScaleFwdParams {
  memory::dims src_dims;
  memory::desc src_md;
  memory::desc dst_md;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  PostOpParam post_op_params;

  MklReorderWithScaleFwdParams(memory::dims src_dims, memory::desc src_md,
                               memory::desc dst_md)
      : src_dims(src_dims), src_md(src_md), dst_md(dst_md) {}
};

class MklReorderWithScalePrimitive : public MklPrimitive {
 public:
  explicit MklReorderWithScalePrimitive(
      const MklReorderWithScaleFwdParams& fwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create reorder primitive
    Setup(fwdParams);
  }

  ~MklReorderWithScalePrimitive() {}

  std::shared_ptr<primitive> GetPrimitive() { return context_.reorder_prim; }

  void Execute(void* src_data, void* dst_data,
               std::shared_ptr<stream> reorder_stream) {
#ifdef DNNL_AARCH64_USE_ACL
    mutex_lock lock(primitive_execution_mu_);
#endif
#ifndef ENABLE_ONEDNN_OPENMP
    context_.src_mem->set_data_handle(src_data, *reorder_stream);
    context_.dst_mem->set_data_handle(dst_data, *reorder_stream);
#else
    context_.src_mem->set_data_handle(src_data);
    context_.dst_mem->set_data_handle(dst_data);
#endif  // !ENABLE_ONEDNN_OPENMP
    context_.reorder_prim->execute(*reorder_stream, context_.prim_args);
    // After execution, set data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

 private:
  // Primitive reuse context for reorder
  struct ReorderContext {
    // MKL-DNN memory
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> dst_mem;

    // Reorder primitive descriptor and primitive
    std::shared_ptr<reorder::primitive_desc> reorder_pd;
    std::shared_ptr<primitive> reorder_prim;

    // Stream and primitive vector
    std::shared_ptr<dnnl::stream> reorder_stream;

    std::unordered_map<int, dnnl::memory> prim_args;

    ReorderContext()
        : src_mem(nullptr),
          dst_mem(nullptr),
          reorder_pd(nullptr),
          reorder_prim(nullptr) {}
  } context_;

  // Reorder primitive setup
  void Setup(const MklReorderWithScaleFwdParams& fwdParams) {
    // Create memory descriptors for reorder data with specified format
    context_.src_mem.reset(
        new memory(fwdParams.src_md, cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(fwdParams.dst_md, cpu_engine_, DummyData));

    // Check if there is any fusion as post-ops
    auto const& post_op_params = fwdParams.post_op_params;
    dnnl::primitive_attr post_ops_attr;

    DCHECK(post_op_params.name == "scale");
    DCHECK_EQ(post_op_params.param.size(), 1);
    std::vector<float> scales;
    scales.push_back(post_op_params.param[0]);
    post_ops_attr.set_output_scales(0, scales);

    context_.reorder_pd.reset(
        new ReorderPd(cpu_engine_, context_.src_mem->get_desc(), cpu_engine_,
                      context_.dst_mem->get_desc(), post_ops_attr));

    // Create reorder primitive
    context_.reorder_prim.reset(new reorder(*context_.reorder_pd));
    context_.prim_args.insert({DNNL_ARG_FROM, *context_.src_mem});
    context_.prim_args.insert({DNNL_ARG_TO, *context_.dst_mem});
  }

#ifdef DNNL_AARCH64_USE_ACL
  mutex primitive_execution_mu_;
#endif
};

template <typename T>
class MklReorderWithScalePrimitiveFactory : public MklPrimitiveFactory<T> {
 public:
  static MklReorderWithScalePrimitive* Get(
      const memory* from, const memory* to,
      const MklReorderWithScaleFwdParams& fwdParams) {
    // Try to find a suitable primitive from the cached pool
    auto reorderPrim = static_cast<MklReorderWithScalePrimitive*>(
        MklReorderWithScalePrimitiveFactory<T>::GetInstance().GetReorder(
            from, to, fwdParams));
    if (reorderPrim == nullptr) {
      reorderPrim = new MklReorderWithScalePrimitive(fwdParams);
      MklReorderWithScalePrimitiveFactory<T>::GetInstance().SetReorder(
          from, to, reorderPrim, fwdParams);
    }
    return reorderPrim;
  }

  static MklReorderWithScalePrimitiveFactory& GetInstance() {
    static MklReorderWithScalePrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklReorderWithScalePrimitiveFactory() {}
  ~MklReorderWithScalePrimitiveFactory() {}

  static string CreateKey(const memory* from, const memory* to,
                          const MklReorderWithScaleFwdParams& fwdParams) {
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(MklReorderPrimitiveFactory<T>::CreateKey(from, to));
    // Generate key for post-op scale
    if (fwdParams.post_op_params.name == "scale") {
      DCHECK_EQ(fwdParams.post_op_params.param.size(), 1);
      key_creator.AddAsKey(fwdParams.post_op_params.name);
      key_creator.AddAsKey(fwdParams.post_op_params.param[0]);
    } else {
      return string("not_a_key");
    }

    return key_creator.GetKey();
  }

  MklPrimitive* GetReorder(const memory* from, const memory* to,
                           const MklReorderWithScaleFwdParams& fwdParams) {
    string key = CreateKey(from, to, fwdParams);
    return this->GetOp(key);
  }

  void SetReorder(const memory* from, const memory* to, MklPrimitive* op,
                  const MklReorderWithScaleFwdParams& fwdParams) {
    string key = CreateKey(from, to, fwdParams);
    this->SetOp(key, op);
  }
};
}  // namespace tensorflow
#endif  // INTEL_MKL
