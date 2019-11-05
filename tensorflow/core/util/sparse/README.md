SparseTensor
============

Sparse Tensors are stored as two dense tensors and a shape:

*  `indices`: a `brain::Tensor` storing a matrix, typically `int64`
*  `values`: a `brain::Tensor` storing a vector with values of type T.
*  `shape`: a `TensorShape` storing the bounds of the underlying tensor
*  `order`: (optional) a `gtl::InlinedVector<int64,8>` with the dimensions
            along which the indices are ordered.

Let

    ix = indices.matrix<int64>()
    vals = values.vec<T>()

The shape of `ix` is `N x NDIMS`, and each row corresponds to the
index of a single element of the sparse tensor.

The length of `vals` must be `N`, and `vals(i)` corresponds to the
value with index `ix(i,:)`.

Shape must be a `TensorShape` with `dims() == NDIMS`.
The shape is the full shape of the dense tensor these indices
represent.

To be specific, the representation (pseudocode) is:

    tensor[ix[i,:]] == vals[i] for i = 0, ..., N-1

Ordering
--------

Indices need not be provided in order.  For example, the following
index matrix is ordered according to dimension order `{0, 1, 2}`.

    [0 0 1]
    [0 1 1]
    [2 0 2]

However, you can provide an unordered version:

    [2 0 2]
    [0 0 1]
    [0 1 1]

If the SparseTensor is constructed without a provided order, then a
the default order is `{-1, ..., -1}`.  Certain operations will fail or crash
when the order is not provided.

Resorting the SparseTensor in-place (which resorts the underlying index and
values tensors in-place) will update the order.  The cost of reordering the
matrix is `O(N*log(N))`, and requires `O(N)` additional temporary space to store
a reordering index.  If the default order is not specified and reordering is not
performed, the following will happen:

*  `group()` will **raise an assertion failure**
*  `IndicesValid()` will **raise an assertion failure**

To update the internal index ordering after construction, call
`Reorder<T>()` via, e.g., `Reorder<T>({0,1,2})`.
After this step, all the above methods should work correctly.

The method `IndicesValid()` checks to make sure:

*  `0 <= ix(i, d) < shape.dim_size(d)`
*  indices do not repeat
*  indices are in order

Iterating
---------

### group({grouping dims})

*  provides an iterator that groups entries according to
   dimensions you care about
*  may require a sort if your data isn't presorted in a way that's
   compatible with grouping_dims
*  for each group, returns the group index (values of the group
   dims for this iteration), the subset of indices in this group,
   and the subset of values in this group.  these are lazy outputs
   so to read them individually, copy them as per the example
   below.

#### **NOTE**
`group({dim0, ..., dimk})` will **raise an assertion failure** if the
order of the SparseTensor does not match the dimensions you wish to group by.
You must either have your indices in the correct order and construct the
SparseTensor with

    order = {dim0, ..., dimk, ...}

or call

    Reorder<T>({dim0, .., dimk, ...})

to sort the SparseTensor before grouping.

Example of grouping:

    Tensor indices(DT_INT64, TensorShape({N, NDIMS});
    Tensor values(DT_STRING, TensorShape({N});
    TensorShape shape({dim0,...});
    SparseTensor sp(indices, vals, shape);
    sp.Reorder<tstring>({1, 2, 0, 3, ...}); // Must provide NDIMS dims.
    // group according to dims 1 and 2
    for (const auto& g : sp.group({1, 2})) {
      cout << "vals of ix[:, 1,2] for this group: "
           << g.group()[0] << ", " << g.group()[1];
      cout << "full indices of group:\n" << g.indices();
      cout << "values of group:\n" << g.values();

      TTypes<int64>::UnalignedMatrix g_ix = g.indices();
      TTypes<tstring>::UnalignedVec g_v = g.values();
      ASSERT(g_ix.dimension(0) == g_v.size());  // number of elements match.
    }


ToDense
--------

Converts sparse tensor to dense.  You must provide a pointer to the
dense tensor (preallocated).  `ToDense()` will optionally
preinitialize the tensor with zeros.

Shape checking is performed, as is boundary checking.

    Tensor indices(DT_INT64, TensorShape({N, NDIMS});
    Tensor values(DT_STRING, TensorShape({N});
    TensorShape shape({dim0,...});
    SparseTensor sp(indices, vals, shape);
    ASSERT(sp.IndicesValid());  // checks ordering & index bounds.

    Tensor dense(DT_STRING, shape);
    // initialize other indices to zero.  copy.
    ASSERT(sp.ToDense<tstring>(&dense, true));


Concat
--------

Concatenates multiple SparseTensors and returns a new SparseTensor.
This concatenation is with respect to the "dense" versions of these
SparseTensors.  Concatenation is performed along dimension order[0]
of all tensors.  As a result, shape[order[0]] may differ across
the inputs, but shape[d] for d != order[0] must match across all inputs.

We call order[0] the **primary dimension**.

**Prerequisites**

*  The inputs' ranks must all match.
*  The inputs' order[0] must all match.
*  The inputs' shapes must all match except for dimension order[0].
*  The inputs' values must all be of the same type.

If any of these are false, concat will die with an assertion failure.

Example:
Concatenate two sparse matrices along columns.

Matrix 1:

    [0 0 1]
    [2 0 0]
    [3 0 4]

Matrix 2:

    [0 0 0 0 0]
    [0 1 0 0 0]
    [2 0 0 1 0]

Concatenated Matrix:

    [0 0 1 0 0 0 0 0]
    [2 0 0 0 1 0 0 0]
    [3 0 4 2 0 0 1 0]

Expected input shapes, orders, and `nnz()`:

    shape_1 = TensorShape({3, 3})
    shape_2 = TensorShape({3, 8})
    order_1 = {1, 0}  // primary order is 1, columns
    order_2 = {1, 0}  // primary order is 1, must match
    nnz_1 = 4
    nnz_2 = 3

Output shapes and orders:

    conc_shape = TensorShape({3, 11})  // primary dim increased, others same
    conc_order = {1, 0}  // Orders match along all inputs
    conc_nnz = 7  // Sum of nonzeros of inputs

Coding Example:

    Tensor ix1(DT_INT64, TensorShape({N1, 3});
    Tensor vals1(DT_STRING, TensorShape({N1, 3});
    Tensor ix2(DT_INT64, TensorShape({N2, 3});
    Tensor vals2(DT_STRING, TensorShape({N2, 3});
    Tensor ix3(DT_INT64, TensorShape({N3, 3});
    Tensor vals3(DT_STRING, TensorShape({N3, 3});

    SparseTensor st1(ix1, vals1, TensorShape({10, 20, 5}), {1, 0, 2});
    SparseTensor st2(ix2, vals2, TensorShape({10, 10, 5}), {1, 0, 2});
    // For kicks, st3 indices are out of order, but order[0] matches so we
    // can still concatenate along this dimension.
    SparseTensor st3(ix3, vals3, TensorShape({10, 30, 5}), {1, 2, 0});

    SparseTensor conc = SparseTensor::Concat<string>({st1, st2, st3});
    Tensor ix_conc = conc.indices();
    Tensor vals_conc = conc.values();
    EXPECT_EQ(conc.nnz(), st1.nnz() + st2.nnz() + st3.nnz());
    EXPECT_EQ(conc.Shape(), TensorShape({10, 60, 5}));
    EXPECT_EQ(conc.Order(), {-1, -1, -1});

    // Reorder st3 so all input tensors have the exact same orders.
    st3.Reorder<tstring>({1, 0, 2});
    SparseTensor conc2 = SparseTensor::Concat<string>({st1, st2, st3});
    EXPECT_EQ(conc2.Order(), {1, 0, 2});
    // All indices' orders matched, so output is in order.
    EXPECT_TRUE(conc2.IndicesValid());
