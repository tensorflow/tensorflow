/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Skipgram")
    .Deprecated(19,
                "Moving word2vec into tensorflow_models/tutorials and "
                "deprecating its ops here as a result")
    .Output("vocab_word: string")
    .Output("vocab_freq: int32")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Attr("min_count: int = 5")
    .Attr("subsample: float = 1e-3")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("NegTrain")
    .Deprecated(19,
                "Moving word2vec into tensorflow_models/tutorials and "
                "deprecating its ops here as a result")
    .Input("w_in: Ref(float)")
    .Input("w_out: Ref(float)")
    .Input("examples: int32")
    .Input("labels: int32")
    .Input("lr: float")
    .SetIsStateful()
    .Attr("vocab_count: list(int)")
    .Attr("num_negative_samples: int")
    .SetShapeFn(shape_inference::UnknownShape);

}  // end namespace tensorflow
