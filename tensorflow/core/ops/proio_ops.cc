#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("ExtractEnergyDeps")
    .Input("events_in: string")
    .Output("energy_depositions: float32")
    .Output("hit_ids: float32")
    .Output("hit_assoc: float32")
    .Output("hit_assoc_inv: float32")
    .Attr("event_tag: string = 'Tracker'")
    .Attr("n_samples: int = 1")
    .Attr("max_n_deps: int = 1024")
    .SetShapeFn([](InferenceContext* c) {
        DimensionHandle eventDim = c->Dim(c->input(0), 0);
        int nSamples;
        TF_RETURN_IF_ERROR(c->GetAttr<int>("n_samples", &nSamples));

        c->set_output(0, c->MakeShape({eventDim, nSamples, c->UnknownDim(), 14}));
        c->set_output(1, c->MakeShape({eventDim, c->UnknownDim()}));
        c->set_output(2, c->MakeShape({eventDim, c->UnknownDim(), c->UnknownDim()}));
        c->set_output(3, c->MakeShape({eventDim, c->UnknownDim(), c->UnknownDim()}));

        return Status::OK();
    });

REGISTER_OP("CompCluster")
    .Input("events_in: string")
    .Input("cross_entropy: float32")
    .Input("entropy: float32")
    .Input("hit_ids: float32")
    .Output("events_out: string")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

}  // namespace tensorflow
