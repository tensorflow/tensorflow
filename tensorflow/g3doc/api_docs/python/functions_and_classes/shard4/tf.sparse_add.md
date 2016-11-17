### `tf.sparse_add(a, b, thresh=0)` {#sparse_add}

Adds two tensors, at least one of each is a `SparseTensor`.

If one `SparseTensor` and one `Output` are passed in, returns an `Output`.  If
both arguments are `SparseTensor`s, this returns a `SparseTensor`.  The order
of arguments does not matter.  Use vanilla `tf.add()` for adding two dense
`Output`s.

The indices of any input `SparseTensor` are assumed ordered in standard
lexicographic order.  If this is not the case, before this step run
`SparseReorder` to restore index ordering.

If both arguments are sparse, we perform "clipping" as follows.  By default,
if two values sum to zero at some index, the output `SparseTensor` would still
include that particular location in its index, storing a zero in the
corresponding value slot.  To override this, callers can specify `thresh`,
indicating that if the sum has a magnitude strictly smaller than `thresh`, its
corresponding value and index would then not be included.  In particular,
`thresh == 0.0` (default) means everything is kept and actual thresholding
happens only for a positive value.

For example, suppose the logical sum of two sparse operands is (densified):

    [       2]
    [.1     0]
    [ 6   -.2]

Then,

    * `thresh == 0` (the default): all 5 index/value pairs will be returned.
    * `thresh == 0.11`: only .1 and 0  will vanish, and the remaining three
        index/value pairs will be returned.
    * `thresh == 0.21`: .1, 0, and -.2 will vanish.

##### Args:


*  <b>`a`</b>: The first operand; `SparseTensor` or `Output`.
*  <b>`b`</b>: The second operand; `SparseTensor` or `Output`.  At least one operand
    must be sparse.
*  <b>`thresh`</b>: A 0-D `Output`.  The magnitude threshold that determines if an
  output value/index pair takes space.  Its dtype should match that of the
  values if they are real; if the latter are complex64/complex128, then the
  dtype should be float32/float64, correspondingly.

##### Returns:

  A `SparseTensor` or an `Output`, representing the sum.

##### Raises:


*  <b>`TypeError`</b>: If both `a` and `b` are `Output`s.  Use `tf.add()` instead.

