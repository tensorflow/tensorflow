# Entropy coder

This module contains range encoder and range decoder which can encode integer
data into string with cumulative distribution functions (CDF).

## Data and CDF values

The data to be encoded should be non-negative integers in half-open interval
`[0, m)`. Then a CDF is represented as an integral vector of length `m + 1`
where `CDF(i) = f(Pr(X < i) * 2^precision)` for i = 0,1,...,m, and `precision`
is an attribute in range `0 < precision <= 16`. The function `f` maps real
values into integers, e.g., round or floor. It is important that to encode a
number `i`, `CDF(i + 1) - CDF(i)` cannot be zero.

Note that we used `Pr(X < i)` not `Pr(X <= i)`, and therefore CDF(0) = 0 always.

## RangeEncode: data shapes and CDF shapes

For each data element, its CDF has to be provided. Therefore if the shape of CDF
should be `data.shape + (m + 1,)` in NumPy-like notation. For example, if `data`
is a 2-D tensor of shape (10, 10) and its elements are in `[0, 64)`, then the
CDF tensor should have shape (10, 10, 65).

This may make CDF tensor too large, and in many applications all data elements
may have the same probability distribution. To handle this, `RangeEncode`
supports limited broadcasting CDF into data. Broadcasting is limited in the
following sense:

- All CDF axes but the last one is broadcasted into data but not the other way
  around,
- The number of CDF axes does not extend, i.e., `CDF.ndim == data.ndim + 1`.

In the previous example where data has shape (10, 10), the following are
acceptable CDF shapes:

- (10, 10, 65)
- (1, 10, 65)
- (10, 1, 65)
- (1, 1, 65)

## RangeDecode

`RangeEncode` encodes neither data shape nor termination character. Therefore
the decoder should know how many characters are encoded into the string, and
`RangeDecode` takes the encoded data shape as the second argument. The same
shape restrictions as `RangeEncode` inputs apply here.

## Example

```python
data = tf.random_uniform((128, 128), 0, 10, dtype=tf.int32)

histogram = tf.bincount(data, minlength=10, maxlength=10)
cdf = tf.cumsum(histogram, exclusive=False)
# CDF should have length m + 1.
cdf = tf.pad(cdf, [[1, 0]])
# CDF axis count must be one more than data.
cdf = tf.reshape(cdf, [1, 1, -1])

# Note that data has 2^14 elements, and therefore the sum of CDF is 2^14.
data = tf.cast(data, tf.int16)
encoded = coder.range_encode(data, cdf, precision=14)
decoded = coder.range_decode(encoded, tf.shape(data), cdf, precision=14)

# data and decoded should be the same.
sess = tf.Session()
x, y = sess.run((data, decoded))
assert np.all(x == y)
```

## Authors
Sung Jin Hwang (github: [ssjhv](https://github.com/ssjhv)) and Nick Johnston
(github: [nmjohn](https://github.com/nmjohn))
