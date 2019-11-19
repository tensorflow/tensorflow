# Tiled layout

Caution: Tiled layout is *pre-release* and this describes how it's intended to
work. Errors may be silently ignored.

<center> ![](images/xla_array_layout_figure1.png)

Figure 1 </center>

Figure 1 shows how an array F32[3,5] is laid out in memory with 2x2 tiling. A
shape with this layout is written as F32[3,5]{1,0:(2,2)}, where 1,0 relates to
the physical order of dimensions (minor_to_major field in Layout) while (2,2)
after the colon indicates tiling of the physical dimensions by a 2x2 tile.

Intuitively tiles are laid out to cover the shape and then within each tile,
elements are then laid out without tiling, as in the example above, where the
right part of the example shows the layout in memory, including the white
padding elements that are added in order to have complete 2x2 tiles even though
the original array bounds are not even.

The extra elements in the padding are not required to contain any particular
value.

## Linear index formulas for tiling given a shape and a tile

Without tiling, an element e=(e<sub>n</sub>, e<sub>n-1</sub>, ... ,
e<sub>1</sub>) in an array with array bounds d=(d<sub>n</sub>, d<sub>n-1</sub>,
... , d<sub>1</sub>) (d1 is the most minor dimension) is laid out by major to
minor order at position:

&nbsp;&nbsp; linear_index(e, d) \
= linear_index((e<sub>n</sub>, e<sub>n-1</sub>, ... , e<sub>1</sub>),
(d<sub>n</sub>, d<sub>n-1</sub>, ... , d<sub>1</sub>)) \
= e<sub>n</sub>d<sub>n-1</sub>...d<sub>1</sub> +
e<sub>n-1</sub>d<sub>n-2</sub>...d<sub>1</sub> + ... + e<sub>1</sub>

For simplicity of notation in this document we assume a tile has the same number
of dimensions as the array. In XLA's implementation of tiling, this is
generalized to tilings with fewer dimensions by leaving the initial most-major
dimensions unchanged and applying the tiling only to the most minor dimensions,
so that the tiling that is specified mentions a suffix of the physical
dimensions of the shape being tiled.

When tiling of size (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>) is
used, an element in the array with indices (e<sub>n</sub>, e<sub>n-1</sub>, ...
, e<sub>1</sub>) is mapped to this position in the final layout:

&nbsp;&nbsp; linear_index_with_tile(e, d, t) \
= linear_index((⌊e/t⌋, e mod t), (⌈d/t⌉, t)) &nbsp; &nbsp; (arithmetic is
elementwise, (a,b) is concatenation) \
= linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... ,
⌊e<sub>1</sub>/t<sub>1</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... ,
e<sub>1</sub> mod t<sub>1</sub>), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... ,
⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, t<sub>n-1</sub>, ... ,
t<sub>1</sub>)) \
= linear_index((⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... ,
⌊e<sub>1</sub>/t<sub>1</sub>⌋), (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... ,
⌈d<sub>1</sub>/t<sub>1</sub>⌉))∙t<sub>n</sub>t<sub>n-1</sub>...t<sub>1</sub> +
linear_index((e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod
t<sub>1</sub>), (t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>))

The layout can be thought of as having two parts:
(⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... , ⌊e<sub>1</sub>/t<sub>1</sub>⌋), which
corresponds to a tile index in an array of tiles of size
(⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... , ⌈d<sub>1</sub>/t<sub>1</sub>⌉), and
(e<sub>n</sub> mod t<sub>n</sub>, ... , e<sub>1</sub> mod t<sub>1</sub>), which
corresponds to a within-tile index. The ceil function appears in
⌈d<sub>i</sub>/t<sub>i</sub>⌉ because if tiles overrun the bounds of the larger
array, padding is inserted as in Figure 1. Both the tiles and elements within
tiles are laid out recursively without tiling.

For the example in Figure 1, element (2,3) has tile index (1,1), and within-tile
index (0,1), for a combined coordinate vector of (1, 1, 0, 1). The tile indices
have bounds (2, 3) and the tile itself is (2, 2) for a combined vector of (2, 3,
2, 2). The linear index with tile for the element with index (2, 3) in the
logical shape is then

&nbsp;&nbsp; linear_index_with_tile((2,3), (3,5), (2,2)) \
= linear_index((1,1,0,1), (2,3,2,2)) \
= linear_index((1,1), (2,3)) ∙ 2 ∙ 2 + linear_index((0,1), (2,2)) \
= (1 ∙ 3 + 1) ∙ 2 ∙ 2 + (0 ∙ 2 + 1) \
= 17.

# Tiling as pad-reshape-transpose

Tiling-based layout operates as follows: \
Consider an array of dimensions (d<sub>n</sub>, d<sub>n-1</sub>, ... , d1) (d1
is the most minor dimension). When it’s laid out with tiling of size
(t<sub>n</sub>, t<sub>n-1</sub>, ... , t<sub>1</sub>) (t<sub>1</sub> is the most
minor dimension), that tiling can be described in terms of pad-reshape-transpose
in the following way.

1.  The array is padded to (⌈d<sub>n</sub>/t<sub>n</sub>⌉∙t<sub>n</sub>, ... ,
    ⌈d<sub>1</sub>/t<sub>1</sub>⌉∙t<sub>1</sub>).
2.  Each dimension i is broken into (⌈d<sub>i</sub>/t</sub>i</sub>⌉,
    t<sub>i</sub>), i.e. the array is reshaped to \
    &nbsp; &nbsp; (⌈d<sub>n</sub>/t<sub>n</sub>⌉, t<sub>n</sub>, ... ,
    ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>1</sub>). \
    There is no physical layout change in this reshape by itself, so this
    reshape is a bitcast. If one is not explicitly thinking of a tiling, this
    reshape could express any shape with the same number of elements as the
    padded shape - the example here is of how to express a tile in this way.
3.  A transpose happens by moving t<sub>n</sub>, ... , t<sub>1</sub> to the most
    minor dimensions while keeping their relative order, so that the order of
    dimensions from most major to most minor becomes \
    &nbsp; &nbsp; (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... ,
    ⌈d<sub>1</sub>/t<sub>1</sub>⌉, t<sub>n</sub>, ... , t<sub>1</sub>).

The final shape has the prefix \
&nbsp; &nbsp; (⌈d<sub>n</sub>/t<sub>n</sub>⌉, ... ,
⌈d<sub>1</sub>/t<sub>1</sub>⌉), which describes the number of tiles in each
dimension. An element in the array (e<sub>n</sub>, ... , e<sub>1</sub>) is
mapped to this element in the final shape: \
&nbsp; &nbsp; (⌊e<sub>n</sub>/t<sub>n</sub>⌋, ... ,
⌊e<sub>0</sub>/t<sub>0</sub>⌋, e<sub>n</sub> mod t<sub>n</sub>, ... ,
e<sub>1</sub> mod t<sub>1</sub>). It is easy to see that the linear index of the
element follows the formula above as expected.

# Repeated tiling

XLA's tiling becomes even more flexible by applying it repeatedly.

<center> ![](images/xla_array_layout_figure2.png)

Figure 2 </center>

Figure 2 shows how an array of size 4x8 is tiled by two levels of tiling (first
2x4 then 2x1). We represent this repeated tiling as (2,4)(2,1). Each color
indicates a 2x4 tile and each red border box is a 2x1 tile. The numbers
indicates the linear index in memory of that element in the tiled format. This
format matches the format used for BF16 on TPU, except that the initial tile is
bigger, namely the tiling is (8,128)(2,1), where the purpose of the second
tiling by 2x1 is to collect together two 16 bit values to form one 32 bit value
in a way that aligns with the architecture of a TPU.

Note that a second or later tile can refer to both the minor within-tile
dimensions, which just rearranges data within the tile, as in this example with
(8,128)(2,1), but can also refer to the major cross-tile dimensions from the
prior tiling.

# Combining dimensions using tiles

XLA's tiling also supports combining dimensions. For example, it can combine
dimensions in F32[2,7,8,11,10]{4,3,2,1,0} into F32[112,110]{1,0} first before
tiling it with (2,3). The tile used is (&lowast;,&lowast;,2,&lowast;,3). Here an
asterisk in a tile implies taking that dimension and combining it with the next
more minor dimension. Multiple adjacent dimensions can be subsumed together into
one dimension. A subsumed dimension is represented by a tile value of -1 in that
dimension of the tile, which is not otherwise valid in a tile as a dimension
size.

More precisely, if dimension i of the shape is eliminated via an asterisk in the
tile, then before the prior definition of tiling is applied, that dimension is
removed from both the shape being tiled and the tile vector, and what was
dimension i-1 of the shape has its array bound increased from d<sub>i-1</sub> to
d<sub>i</sub>d<sub>i-1</sub>. This step is repeated for each asterisk in the
tile vector.
