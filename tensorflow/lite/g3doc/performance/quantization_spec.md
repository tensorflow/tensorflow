# TensorFlow Lite 8-bit quantization specification

### Specification summary

8-bit quantization approximates floating point values using the following
formula.

$$real\_value = (int8\_value - zero\_point) \times scale$$

Per-axis (aka per-channel) or per-layer weights are represented by `int8` two’s
complement values in the range `[-127, 127]` with zero-point equal to 0.
Per-layer activations/inputs are represented by `int8` two’s complement values
in the range `[-128, 127]`, with a zero-point in range `[-128, 127]`.

There are other exceptions for particular operations that are documented below.

Note: In the past our quantized tooling used per-layer, asymmetric, `uint8`
quantization. New tooling, reference kernels, and optimized kernels for 8-bit
quantization will use this spec.

### Signed integer vs unsigned integer

TensorFlow Lite quantization will primarily prioritize tooling and kernels for
`int8` quantization for 8-bit. This is for the convenience of symmetric
quantization being represented by zero-point equal to 0. Additionally many
backends have additional optimizations for `int8xint8` accumulation.

### Per-axis vs per-layer

Per-layer quantization means that there will be one scale and/or zero-point per
entire tensor. Per-axis quantization means that there will be one scale and/or
`zero_point` per slice in the `quantized_dimension`. The quantized dimension
specifies the dimension of the Tensor's shape that the scales and zero-points
correspond to. For example, a tensor `t`, with `dims=[4, 3, 2, 1]` with
quantization params: `scale=[1.0, 2.0, 3.0]`, `zero_point=[1, 2, 3]`,
`quantization_dimension=1` will be quantized across the second dimension of t:

    t[:, 0, :, :] will have scale[0]=1.0, zero_point[0]=1
    t[:, 1, :, :] will have scale[1]=2.0, zero_point[0]=2
    t[:, 2, :, :] will have scale[2]=3.0, zero_point[0]=3

Often, the quantized_dimension is the output_channel of the weights of
convolutions, but in theory it can be the dimension that corresponds to each
dot-product in the kernel implementation, allowing more quantization granularity
without performance implications. This has large improvements to accuracy.

TFLite has per-axis support for a growing number of operations. At the time of
this document support exists for Conv2d and DepthwiseConv2d.

### Symmetric vs asymmetric

Activations are asymmetric: they can have their zero-point anywhere within the
signed `int8` range `[-128, 127]`. Many activations are asymmetric in nature and
a zero-point is an relatively inexpensive way to effectively get up to an extra
binary bit of precision. Since activations are only multiplied by constant
weights, the constant zero-point value can be optimized pretty heavily.

Weights are symmetric: forced to have zero-point equal to 0. Weight values are
multiplied by dynamic input and activation values. This means that there is a
unavoidable runtime cost of multiplying the zero-point of the weight with the
activation value. By enforcing that zero-point is 0 we can avoid this cost.

Explanation of the math:

$$A$$ is a $$m \times n$$ matrix of quantized activations. <br />
$$B$$ is a $$n \times p$$ matrix of quantized weights. <br />
Consider multiplying the $$j$$th row of $$A$$, $$a_j$$ by the $$k$$th row of
$$B$$, $$b_k$$, both of length $$n$$. The quantized integer values and
zero-points values are $$q_a$$, $$z_a$$ and $$q_b$$, $$q_b$$ respectively.

$$a_j \cdot b_k = \sum_{i=0}^{n} a_{j}^{(i)} b_{k}^{(i)} =
\sum_{i=0}^{n} (q_{a}^{(i)} - z_a) (q_{b}^{(i)} - z_b) =
\sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)} - \sum_{i=0}^{n} q_{a}^{(i)} z_b -
\sum_{i=0}^{n} q_{b}^{(i)} z_a + \sum_{i=0}^{n} z_a z_b$$

The $$\sum_{i=0}^{n} q_{a}^{(i)} q_{b}^{(i)}$$ term is unavoidable since it’s
performing the dot product of the input value and the weight value.

The $$\sum_{i=0}^{n} q_{b}^{(i)} z_a + \sum_{i=0}^{n} z_a z_b$$ terms are made
up of constants that remain the same per inference invocation, and thus can be
pre-calculated.

The $$\sum_{i=0}^{n} q_{a}^{(i)} z_b$$ term needs to be computed every inference
since the activation changes every inference. By enforcing weights to be
symmetric we can remove the cost of this term.
