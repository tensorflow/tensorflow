# Changes since last release

## Breaking changes to the API

* `AdjustContrast` kernel deprecated, new kernel `AdjustContrastv2` takes and
  outputs float only. `adjust_contrast` now takes all data types.
* `adjust_brightness`'s `delta` argument is now always assumed to be in `[0,1]`
  (as is the norm for images in floating point formats), independent of the
  data type of the input image.
* The image processing ops do not take `min` and `max` inputs any more, casting
  safety is handled by `saturate_cast`, which makes sure over- and underflows
  are handled before casting to data types with smaller ranges.
* For C++ API users: `IsLegacyScalar` and `IsLegacyVector` are now gone from
  `TensorShapeUtils` since TensorFlow is scalar strict within Google (for
  example, the shape argument to `tf.reshape` can't be a scalar anymore).  The
  open source release was already scalar strict, so outside Google `IsScalar`
  and `IsVector` are exact replacements.

## Bug fixes

* The Python API will now properly set the `list` member of `AttrValue` in
  constructed `GraphDef` messages for empty lists.  The serialization of some
  graphs will change, but the change is both forwards and backwards compatible.
  It will break tests that compare a generated `GraphDef` to a golden serialized
  `GraphDef`.


# Release 0.6.0

## Major Features and Improvements

* Python 3.3+ support via changes to python codebase and ability
  to specify python version via ./configure.

* Some improvements to GPU performance and memory usage:
  [convnet benchmarks](https://github.com/soumith/convnet-benchmarks/issues/66)
  roughly equivalent with native cudnn v2 performance.  Improvements mostly due
  to moving to 32-bit indices, faster shuffling kernels.  More improvements to
  come in later releases.


## Bug fixes

* Lots of fixes to documentation and tutorials, many contributed
  by the public.

* 271 closed issues on github issues.

## Backwards-incompatible changes

* tf.nn.fixed_unigram_candidate_sampler changed its default 'distortion'
  attribute from 0.0 to 1.0. This was a bug in the original release
  that is now fixed.

# Release 0.5.0

Initial release of TensorFlow.
