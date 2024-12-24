========================================================== FAILURES ===========================================================
____________________________________________ LaxBackedNumpyTests.testConvolutions2 ____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testConvolutions2>, xshape = (6,), yshape = (12,)
dtype = <class 'ml_dtypes.bfloat16'>, mode = 'valid', op = 'correlate'

    @jtu.sample_product(
      mode=['full', 'same', 'valid'],
      op=['convolve', 'correlate'],
      dtype= float_dtypes, #number_dtypes,
      xshape=one_dim_array_shapes,
      yshape=one_dim_array_shapes,
    )
    def testConvolutions(self, xshape, yshape, dtype, mode, op):
      jnp_op = getattr(jnp, op)
      np_op = getattr(np, op)
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
      precision = lax.Precision.HIGHEST if jtu.test_device_matches(["tpu"]) else None
      jnp_fun = partial(jnp_op, mode=mode, precision=precision)
      def np_fun(x, y):
        return np_op(x, y, mode=mode).astype(dtypes.to_inexact_dtype(dtype))
      tol = {np.float16: 2e-1, np.float32: 1e-1, np.float64: 1e-14,
             np.complex128: 1e-14}
>     self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, tol=tol)

lax_stablehlo_reference_test.py:2065: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1344: in _CheckAgainstNumpy
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f02f239e7a0>, array([  3.    ,  11.375 ,  21.5   ,  -4.6875,  43.25  , -19.875 ,
        -8.9375], dtype=float32), array([  2.90625,  11.375  ,  21.5    ,  -4.625  ,  43.25   , -19.875  ,
        -9.     ], dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.01, atol=0.01', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.01
E           
E           Mismatched elements: 2 / 7 (28.6%)
E           Max absolute difference among violations: 0.09375
E           Max relative difference among violations: 0.03225806
E            ACTUAL: array([  3.    ,  11.375 ,  21.5   ,  -4.6875,  43.25  , -19.875 ,
E                   -8.9375], dtype=float32)
E            DESIRED: array([  2.90625,  11.375  ,  21.5    ,  -4.625  ,  43.25   , -19.875  ,
E                   -9.     ], dtype=float32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________ LaxBackedNumpyTests.testConvolutions4 ____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testConvolutions4>, xshape = (12,), yshape = (6,)
dtype = <class 'ml_dtypes.bfloat16'>, mode = 'same', op = 'correlate'

    @jtu.sample_product(
      mode=['full', 'same', 'valid'],
      op=['convolve', 'correlate'],
      dtype= float_dtypes, #number_dtypes,
      xshape=one_dim_array_shapes,
      yshape=one_dim_array_shapes,
    )
    def testConvolutions(self, xshape, yshape, dtype, mode, op):
      jnp_op = getattr(jnp, op)
      np_op = getattr(np, op)
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
      precision = lax.Precision.HIGHEST if jtu.test_device_matches(["tpu"]) else None
      jnp_fun = partial(jnp_op, mode=mode, precision=precision)
      def np_fun(x, y):
        return np_op(x, y, mode=mode).astype(dtypes.to_inexact_dtype(dtype))
      tol = {np.float16: 2e-1, np.float32: 1e-1, np.float64: 1e-14,
             np.complex128: 1e-14}
>     self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, tol=tol)

lax_stablehlo_reference_test.py:2065: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1344: in _CheckAgainstNumpy
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f02eaf26200>, array([-30.875     , -43.        , -56.25      , -40.5       ,
       -34.5       ,  -9.5       ,   0.96484375,   4.625     ,
        14.3125    , -36.75      ,  20.375     , -65.        ],
      dtype=float32), array([-30.75  , -43.    , -56.5   , -40.5   , -34.5   ,  -9.3125,
         1.    ,   4.5   ,  14.1875, -36.75  ,  20.5   , -65.    ],
      dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.01, atol=0.01', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.01
E           
E           Mismatched elements: 3 / 12 (25%)
E           Max absolute difference among violations: 0.1875
E           Max relative difference among violations: 0.03515625
E            ACTUAL: array([-30.875   , -43.      , -56.25    , -40.5     , -34.5     ,
E                   -9.5     ,   0.964844,   4.625   ,  14.3125  , -36.75    ,
E                   20.375   , -65.      ], dtype=float32)
E            DESIRED: array([-30.75  , -43.    , -56.5   , -40.5   , -34.5   ,  -9.3125,
E                    1.    ,   4.5   ,  14.1875, -36.75  ,  20.5   , -65.    ],
E                 dtype=float32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________ LaxBackedNumpyTests.testConvolutions9 ____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testConvolutions9>, xshape = (12,), yshape = (12,)
dtype = <class 'ml_dtypes.bfloat16'>, mode = 'full', op = 'convolve'

    @jtu.sample_product(
      mode=['full', 'same', 'valid'],
      op=['convolve', 'correlate'],
      dtype= float_dtypes, #number_dtypes,
      xshape=one_dim_array_shapes,
      yshape=one_dim_array_shapes,
    )
    def testConvolutions(self, xshape, yshape, dtype, mode, op):
      jnp_op = getattr(jnp, op)
      np_op = getattr(np, op)
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
      precision = lax.Precision.HIGHEST if jtu.test_device_matches(["tpu"]) else None
      jnp_fun = partial(jnp_op, mode=mode, precision=precision)
      def np_fun(x, y):
        return np_op(x, y, mode=mode).astype(dtypes.to_inexact_dtype(dtype))
      tol = {np.float16: 2e-1, np.float32: 1e-1, np.float64: 1e-14,
             np.complex128: 1e-14}
>     self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, tol=tol)

lax_stablehlo_reference_test.py:2065: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1344: in _CheckAgainstNumpy
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f02eaf26660>, array([  0.05688477,  15.3125    , -22.125     ,  19.125     ,
       -22.125     ,  49.25      , -42.        ,  54.75      ,
       -36.        ,  15.9375    , -22.625     ,  20.625     ,
       -33.        ,  22.25      , -37.        ,  18.375     ,
       -11.3125    ,   4.46875   ,   4.9375    ,   6.78125   ,
        -8.375     ,   5.6875    ,  -1.1875    ], dtype=float32), array([  0.05688477,  15.3125    , -22.25      ,  19.125     ,
       -22.125     ,  49.25      , -42.        ,  54.75      ,
       -36.        ,  15.875     , -22.5       ,  20.625     ,
       -33.25      ,  22.5       , -36.75      ,  18.375     ,
       -11.3125    ,   4.4375    ,   4.96875   ,   6.75      ,
        -8.375     ,   5.6875    ,  -1.1875    ], dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.01, atol=0.01', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.01
E           
E           Mismatched elements: 1 / 23 (4.35%)
E           Max absolute difference among violations: 0.25
E           Max relative difference among violations: 0.01111111
E            ACTUAL: array([  0.056885,  15.3125  , -22.125   ,  19.125   , -22.125   ,
E                   49.25    , -42.      ,  54.75    , -36.      ,  15.9375  ,
E                  -22.625   ,  20.625   , -33.      ,  22.25    , -37.      ,...
E            DESIRED: array([  0.056885,  15.3125  , -22.25    ,  19.125   , -22.125   ,
E                   49.25    , -42.      ,  54.75    , -36.      ,  15.875   ,
E                  -22.5     ,  20.625   , -33.25    ,  22.5     , -36.75    ,...

/usr/lib/python3.11/contextlib.py:81: AssertionError
__________________________________ LaxBackedNumpyTests.testConvolutionsPreferredElementType4 __________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testConvolutionsPreferredElementType4>, xshape = (12,)
yshape = (6,), dtype = <class 'ml_dtypes.bfloat16'>, mode = 'same', op = 'correlate'

    @jtu.sample_product(
      mode=['full', 'same', 'valid'],
      op=['convolve', 'correlate'],
      dtype=float_dtypes, #number_dtypes,
      xshape=one_dim_array_shapes,
      yshape=one_dim_array_shapes,
    )
    @jtu.skip_on_devices("cuda", "rocm")  # backends don't support all dtypes.
    def testConvolutionsPreferredElementType(self, xshape, yshape, dtype, mode, op):
      jnp_op = getattr(jnp, op)
      np_op = getattr(np, op)
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
      precision = lax.Precision.HIGHEST if jtu.test_device_matches(["tpu"]) else None
      jnp_fun = partial(jnp_op, mode=mode, precision=precision,
                        preferred_element_type=dtype)
      def np_fun(x, y):
        return np_op(x, y, mode=mode).astype(dtype)
      tol = {np.float16: 2e-1, np.float32: 1e-2, np.float64: 1e-14,
             np.complex128: 1e-14}
>     self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, tol=tol)

lax_stablehlo_reference_test.py:2088: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1344: in _CheckAgainstNumpy
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f02eaf6d620>, array([  0.8828125, -29.125    , -12.875    ,  28.625    ,  18.625    ,
       -14.5625   ,   1.59375  , -25.75     ,  -1.4296875,  20.125    ,
        10.5      , -13.125    ], dtype=float32), array([  0.875   , -29.125   , -12.9375  ,  28.75    ,  18.5     ,
       -14.5625  ,   1.46875 , -25.875   ,  -1.453125,  20.      ,
        10.4375  , -13.125   ], dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.01, atol=0.01', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.01
E           
E           Mismatched elements: 1 / 12 (8.33%)
E           Max absolute difference among violations: 0.125
E           Max relative difference among violations: 0.08510638
E            ACTUAL: array([  0.882812, -29.125   , -12.875   ,  28.625   ,  18.625   ,
E                  -14.5625  ,   1.59375 , -25.75    ,  -1.429688,  20.125   ,
E                   10.5     , -13.125   ], dtype=float32)
E            DESIRED: array([  0.875   , -29.125   , -12.9375  ,  28.75    ,  18.5     ,
E                  -14.5625  ,   1.46875 , -25.875   ,  -1.453125,  20.      ,
E                   10.4375  , -13.125   ], dtype=float32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
__________________________________ LaxBackedNumpyTests.testConvolutionsPreferredElementType9 __________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testConvolutionsPreferredElementType9>, xshape = (12,)
yshape = (12,), dtype = <class 'ml_dtypes.bfloat16'>, mode = 'full', op = 'convolve'

    @jtu.sample_product(
      mode=['full', 'same', 'valid'],
      op=['convolve', 'correlate'],
      dtype=float_dtypes, #number_dtypes,
      xshape=one_dim_array_shapes,
      yshape=one_dim_array_shapes,
    )
    @jtu.skip_on_devices("cuda", "rocm")  # backends don't support all dtypes.
    def testConvolutionsPreferredElementType(self, xshape, yshape, dtype, mode, op):
      jnp_op = getattr(jnp, op)
      np_op = getattr(np, op)
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(xshape, dtype), rng(yshape, dtype)]
      precision = lax.Precision.HIGHEST if jtu.test_device_matches(["tpu"]) else None
      jnp_fun = partial(jnp_op, mode=mode, precision=precision,
                        preferred_element_type=dtype)
      def np_fun(x, y):
        return np_op(x, y, mode=mode).astype(dtype)
      tol = {np.float16: 2e-1, np.float32: 1e-2, np.float64: 1e-14,
             np.complex128: 1e-14}
>     self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True, tol=tol)

lax_stablehlo_reference_test.py:2088: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1344: in _CheckAgainstNumpy
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f02eaf6ca40>, array([ -0.31640625,   1.1015625 ,  -4.1875    ,  -3.265625  ,
         6.375     ,  16.375     ,   9.        , -26.75      ,
       -10.8125    ,  24.5       ,  10.4375    , -15.625     ,
       -11.125     ,   5.09375   ,  -0.53125   , -19.875     ,
       -11.125     ,   0.4609375 ,   2.8125    , -12.625     ,
       -19.625     , -10.6875    ,  -2.8125    ], dtype=float32), array([ -0.31640625,   1.09375   ,  -4.21875   ,  -3.25      ,
         6.375     ,  16.375     ,   9.        , -26.875     ,
       -10.8125    ,  24.5       ,  10.4375    , -15.625     ,
       -11.125     ,   5.28125   ,  -0.546875  , -20.        ,
       -11.125     ,   0.46875   ,   2.828125  , -12.5625    ,
       -19.5       , -10.75      ,  -2.8125    ], dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.01, atol=0.01', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.01
E           
E           Mismatched elements: 2 / 23 (8.7%)
E           Max absolute difference among violations: 0.1875
E           Max relative difference among violations: 0.03550296
E            ACTUAL: array([ -0.316406,   1.101562,  -4.1875  ,  -3.265625,   6.375   ,
E                   16.375   ,   9.      , -26.75    , -10.8125  ,  24.5     ,
E                   10.4375  , -15.625   , -11.125   ,   5.09375 ,  -0.53125 ,...
E            DESIRED: array([ -0.316406,   1.09375 ,  -4.21875 ,  -3.25    ,   6.375   ,
E                   16.375   ,   9.      , -26.875   , -10.8125  ,  24.5     ,
E                   10.4375  , -15.625   , -11.125   ,   5.28125 ,  -0.546875,...

/usr/lib/python3.11/contextlib.py:81: AssertionError
__________________________________________ LaxBackedNumpyTests.testFlatNonzeroSize3 ___________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testFlatNonzeroSize3>, shape = (2, 1, 4)
dtype = <class 'numpy.uint32'>, size = 1, fill_value = (-1,)

    @jtu.sample_product(
      shape=nonempty_array_shapes,
      dtype=all_dtypes,
      fill_value=[None, -1, 10, (-1,), (10,)],
      size=[1, 5, 10],
    )
    def testFlatNonzeroSize(self, shape, dtype, size, fill_value):
      rng = jtu.rand_some_zero(self.rng())
      args_maker = lambda: [rng(shape, dtype)]
      @jtu.ignore_warning(category=DeprecationWarning, message="Calling nonzero on 0d arrays.*")
      def np_fun(x):
        result = np.flatnonzero(x)
        if size <= len(result):
          return result[:size]
        else:
          fill_val = fill_value or 0
          return np.concatenate([result, np.full(size - len(result), fill_val, result.dtype)])
      jnp_fun = lambda x: jnp.flatnonzero(x, size=size, fill_value=fill_value)
      self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)
>     self._CompileAndCheck(jnp_fun, args_maker)

lax_stablehlo_reference_test.py:372: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1322: in _CompileAndCheck
    self.assertAllClose(python_ans, monitored_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f02ca2cbd80>, array([2], dtype=int32), array([-1], dtype=int32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-07, atol=0', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-07, atol=0
E           
E           Mismatched elements: 1 / 1 (100%)
E           Max absolute difference among violations: 3
E           Max relative difference among violations: 3.
E            ACTUAL: array([2], dtype=int32)
E            DESIRED: array([-1], dtype=int32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
________________________________________ LaxBackedNumpyTests.testIntegerPowerOverflow1 ________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testIntegerPowerOverflow1>, x = -1, y = 128

    @jtu.sample_product(
      x=[-1, 0, 1],
      y=[0, 32, 64, 128],
    )
    def testIntegerPowerOverflow(self, x, y):
      # Regression test for https://github.com/jax-ml/jax/issues/5987
      args_maker = lambda: [x, y]
      self._CheckAgainstNumpy(np.power, jnp.power, args_maker)
>     self._CompileAndCheck(jnp.power, args_maker)

lax_stablehlo_reference_test.py:1433: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1322: in _CompileAndCheck
    self.assertAllClose(python_ans, monitored_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f02ae15b1a0>, array(1, dtype=int32), array(-1, dtype=int32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-07, atol=0', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-07, atol=0
E           
E           Mismatched elements: 1 / 1 (100%)
E           Max absolute difference among violations: 2
E           Max relative difference among violations: 2.
E            ACTUAL: array(1, dtype=int32)
E            DESIRED: array(-1, dtype=int32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
_______________________________________________ LaxBackedNumpyTests.testIsClose _______________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testIsClose>

    def testIsClose(self):
      c_isclose = jax.jit(jnp.isclose)
      c_isclose_nan = jax.jit(partial(jnp.isclose, equal_nan=True))
      n = 2
    
      rng = self.rng()
      x = rng.randn(n, 1)
      y = rng.randn(n, 1)
      inf = np.asarray(n * [np.inf]).reshape([n, 1])
      nan = np.asarray(n * [np.nan]).reshape([n, 1])
      args = [x, y, inf, -inf, nan]
    
      for arg0 in args:
        for arg1 in args:
          result_np = np.isclose(arg0, arg1)
          result_jax = jnp.isclose(arg0, arg1)
          result_jit = c_isclose(arg0, arg1)
          self.assertTrue(jnp.all(jnp.equal(result_np, result_jax)))
          self.assertTrue(jnp.all(jnp.equal(result_np, result_jit)))
          result_np = np.isclose(arg0, arg1, equal_nan=True)
          result_jax = jnp.isclose(arg0, arg1, equal_nan=True)
          result_jit = c_isclose_nan(arg0, arg1)
          self.assertTrue(jnp.all(jnp.equal(result_np, result_jax)))
          self.assertTrue(jnp.all(jnp.equal(result_np, result_jit)))
    
      self.assertEqual(np.isclose(6, 10, rtol=0.5), jnp.isclose(6, 10, rtol=0.5))
      key = jax.random.key(0)
>     self.assertTrue(jnp.isclose(key, key))

lax_stablehlo_reference_test.py:3545: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/pjit.py:356: in cache_miss
    outs, out_flat, out_tree, args_flat, jaxpr, attrs_tracked = _python_pjit_helper(
env/lib/python3.11/site-packages/jax/_src/pjit.py:189: in _python_pjit_helper
    out_flat = pjit_p.bind(*args_flat, **p.params)
env/lib/python3.11/site-packages/jax/_src/core.py:2781: in bind
    return self.bind_with_trace(top_trace, args, params)
env/lib/python3.11/site-packages/jax/_src/core.py:442: in bind_with_trace
    out = trace.process_primitive(self, map(trace.full_raise, args), params)
env/lib/python3.11/site-packages/jax/_src/core.py:948: in process_primitive
    return primitive.impl(*tracers, **params)
env/lib/python3.11/site-packages/jax/_src/pjit.py:1764: in _pjit_call_impl
    return xc._xla.pjit(
env/lib/python3.11/site-packages/jax/_src/pjit.py:1739: in call_impl_cache_miss
    out_flat, compiled = _pjit_call_impl_python(
env/lib/python3.11/site-packages/jax/_src/pjit.py:1669: in _pjit_call_impl_python
    ).compile(compile_options)
env/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:2315: in compile
    executable = UnloadedMeshExecutable.from_hlo(
env/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:2829: in from_hlo
    xla_executable = _cached_compilation(
env/lib/python3.11/site-packages/jax/_src/interpreters/pxla.py:2641: in _cached_compilation
    xla_executable = compiler.compile_or_get_cached(
env/lib/python3.11/site-packages/jax/_src/compiler.py:314: in compile_or_get_cached
    return backend_compile(backend, computation, compile_options,
env/lib/python3.11/site-packages/jax/_src/profiler.py:333: in wrapper
    return func(*args, **kwargs)
env/lib/python3.11/site-packages/jax/_src/compiler.py:273: in backend_compile
    raise e
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

backend = <jaxlib.xla_extension.Client object at 0x7f035bd50e80>
module = <jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7f02b42f29b0>
options = <jaxlib.xla_extension.CompileOptions object at 0xd9282420>, host_callbacks = ()

    @profiler.annotate_function
    def backend_compile(
        backend: xc.Client,
        module: ir.Module,
        options: xc.CompileOptions,
        host_callbacks: Sequence[Any],
    ) -> xc.LoadedExecutable:
      # Convert ir.Module to a string representation, unless the backend
      # explicitly flags the ability to handle a module directly (avoiding the
      # overhead of back and forth conversions).
      # TODO(slebedev): Change the backend.compile() to accept ir.Module.
      built_c: Any
      if getattr(backend, "needs_str_ir", True):
        built_c = mlir.module_to_bytecode(module)
      else:
        built_c = module
    
      try:
        # we use a separate function call to ensure that XLA compilation appears
        # separately in Python profiling results
        if host_callbacks:
          return backend.compile(
              built_c, compile_options=options, host_callbacks=host_callbacks
          )
        # Some backends don't have `host_callbacks` option yet
        # TODO(sharadmv): remove this fallback when all backends allow `compile`
        # to take in `host_callbacks`
>       return backend.compile(built_c, compile_options=options)
E       jaxlib.xla_extension.XlaRuntimeError: UNIMPLEMENTED: Unsupported op: %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{replicated}"} : (tensor<2xui32>) -> tensor<2xui32>

env/lib/python3.11/site-packages/jax/_src/compiler.py:267: XlaRuntimeError
__________________________________________ LaxBackedNumpyTests.testNonScalarRepeats1 __________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testNonScalarRepeats1>, fixed_size = False

    @jtu.sample_product(fixed_size=[False, True])
    def testNonScalarRepeats(self, fixed_size):
      '''
      Following numpy test suite from `test_repeat` at
      https://github.com/numpy/numpy/blob/main/numpy/core/tests/test_multiarray.py
      '''
      tol = 1e-5
    
      def test_single(m, args_maker, repeats, axis):
        lax_ans = jnp.repeat(m, repeats, axis)
        numpy_ans = np.repeat(m, repeats, axis)
    
        self.assertAllClose(lax_ans, numpy_ans, rtol=tol, atol=tol)
        if fixed_size:
    
          # Calculate expected size of the repeated axis.
          rep_length = np.repeat(np.zeros_like(m), repeats, axis).shape[axis or 0]
          jnp_fun = lambda arg, rep: jnp.repeat(
              arg, repeats=rep, axis=axis, total_repeat_length=rep_length)
        else:
          jnp_fun = lambda arg: jnp.repeat(arg, repeats = repeats, axis=axis)
        self._CompileAndCheck(jnp_fun, args_maker)
    
      m = jnp.array([1,2,3,4,5,6])
      if fixed_size:
        args_maker = lambda: [m, repeats]
      else:
        args_maker = lambda: [m]
    
      for repeats in [2, jnp.array([1,3,0,1,1,2]), jnp.array([1,3,2,1,1,2]), jnp.array([2])]:
>       test_single(m, args_maker, repeats, axis=None)

lax_stablehlo_reference_test.py:2008: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
lax_stablehlo_reference_test.py:1999: in test_single
    self._CompileAndCheck(jnp_fun, args_maker)
env/lib/python3.11/site-packages/jax/_src/test_util.py:1322: in _CompileAndCheck
    self.assertAllClose(python_ans, monitored_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f029ffc76a0>, array([1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=int32), array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=int32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-07, atol=0', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-07, atol=0
E           
E           Mismatched elements: 8 / 12 (66.7%)
E           Max absolute difference among violations: 3
E           Max relative difference among violations: 1.5
E            ACTUAL: array([1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=int32)
E            DESIRED: array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=int32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________ LaxBackedNumpyTests.testNonzeroSize3 _____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testNonzeroSize3>, shape = ()
dtype = <class 'numpy.int16'>, size = 5, fill_value = (1,)

    @jtu.sample_product(
      [dict(shape=shape, fill_value=fill_value)
        for shape in nonempty_array_shapes
        for fill_value in [None, -1, shape or (1,)]
       ],
      dtype=all_dtypes,
      size=[1, 5, 10],
    )
    def testNonzeroSize(self, shape, dtype, size, fill_value):
      rng = jtu.rand_some_zero(self.rng())
      args_maker = lambda: [rng(shape, dtype)]
      def np_fun(x):
        result = np.nonzero(x)
        if size <= len(result[0]):
          return tuple(arg[:size] for arg in result)
        else:
          fillvals = fill_value if np.ndim(fill_value) else len(result) * [fill_value or 0]
          return tuple(np.concatenate([arg, np.full(size - len(arg), fval, arg.dtype)])
                       for fval, arg in safe_zip(fillvals, result))
      jnp_fun = lambda x: jnp.nonzero(x, size=size, fill_value=fill_value)
      with jtu.ignore_warning(category=DeprecationWarning,
                              message="Calling nonzero on 0d arrays.*"):
>       self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

lax_stablehlo_reference_test.py:336: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1342: in _CheckAgainstNumpy
    lax_ans = lax_op(*args)
lax_stablehlo_reference_test.py:333: in <lambda>
    jnp_fun = lambda x: jnp.nonzero(x, size=size, fill_value=fill_value)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    def nonzero(a: ArrayLike, *, size: int | None = None,
                fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None
        ) -> tuple[Array, ...]:
      """Return indices of nonzero elements of an array.
    
      JAX implementation of :func:`numpy.nonzero`.
    
      Because the size of the output of ``nonzero`` is data-dependent, the function
      is not compatible with JIT and other transformations. The JAX version adds
      the optional ``size`` argument which must be specified statically for
      ``jnp.nonzero`` to be used within JAX's transformations.
    
      Args:
        a: N-dimensional array.
        size: optional static integer specifying the number of nonzero entries to
          return. If there are more nonzero elements than the specified ``size``,
          then indices will be truncated at the end. If there are fewer nonzero
          elements than the specified size, then indices will be padded with
          ``fill_value``, which defaults to zero.
        fill_value: optional padding value when ``size`` is specified. Defaults to 0.
    
      Returns:
        Tuple of JAX Arrays of length ``a.ndim``, containing the indices of each
        nonzero value.
    
      See also:
        - :func:`jax.numpy.flatnonzero`
        - :func:`jax.numpy.where`
    
      Examples:
    
        One-dimensional array returns a length-1 tuple of indices:
    
        >>> x = jnp.array([0, 5, 0, 6, 0, 7])
        >>> jnp.nonzero(x)
        (Array([1, 3, 5], dtype=int32),)
    
        Two-dimensional array returns a length-2 tuple of indices:
    
        >>> x = jnp.array([[0, 5, 0],
        ...                [6, 0, 7]])
        >>> jnp.nonzero(x)
        (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))
    
        In either case, the resulting tuple of indices can be used directly to extract
        the nonzero values:
    
        >>> indices = jnp.nonzero(x)
        >>> x[indices]
        Array([5, 6, 7], dtype=int32)
    
        The output of ``nonzero`` has a dynamic shape, because the number of returned
        indices depends on the contents of the input array. As such, it is incompatible
        with JIT and other JAX transformations:
    
        >>> x = jnp.array([0, 5, 0, 6, 0, 7])
        >>> jax.jit(jnp.nonzero)(x)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[].
        The size argument of jnp.nonzero must be statically specified to use jnp.nonzero within JAX transformations.
    
        This can be addressed by passing a static ``size`` parameter to specify the
        desired output shape:
    
        >>> nonzero_jit = jax.jit(jnp.nonzero, static_argnames='size')
        >>> nonzero_jit(x, size=3)
        (Array([1, 3, 5], dtype=int32),)
    
        If ``size`` does not match the true size, the result will be either truncated or padded:
    
        >>> nonzero_jit(x, size=2)  # size < 3: indices are truncated
        (Array([1, 3], dtype=int32),)
        >>> nonzero_jit(x, size=5)  # size > 3: indices are padded with zeros.
        (Array([1, 3, 5, 0, 0], dtype=int32),)
    
        You can specify a custom fill value for the padding using the ``fill_value`` argument:
    
        >>> nonzero_jit(x, size=5, fill_value=len(x))
        (Array([1, 3, 5, 6, 6], dtype=int32),)
      """
      util.check_arraylike("nonzero", a)
      arr = asarray(a)
      del a
      if ndim(arr) == 0:
>       raise ValueError("Calling nonzero on 0d arrays is not allowed. "
                         "Use jnp.atleast_1d(scalar).nonzero() instead.")
E       ValueError: Calling nonzero on 0d arrays is not allowed. Use jnp.atleast_1d(scalar).nonzero() instead.

env/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py:3469: ValueError
____________________________________________ LaxBackedNumpyTests.testNonzeroSize5 _____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testNonzeroSize5>, shape = (), dtype = <class 'numpy.bool'>
size = 1, fill_value = -1

    @jtu.sample_product(
      [dict(shape=shape, fill_value=fill_value)
        for shape in nonempty_array_shapes
        for fill_value in [None, -1, shape or (1,)]
       ],
      dtype=all_dtypes,
      size=[1, 5, 10],
    )
    def testNonzeroSize(self, shape, dtype, size, fill_value):
      rng = jtu.rand_some_zero(self.rng())
      args_maker = lambda: [rng(shape, dtype)]
      def np_fun(x):
        result = np.nonzero(x)
        if size <= len(result[0]):
          return tuple(arg[:size] for arg in result)
        else:
          fillvals = fill_value if np.ndim(fill_value) else len(result) * [fill_value or 0]
          return tuple(np.concatenate([arg, np.full(size - len(arg), fval, arg.dtype)])
                       for fval, arg in safe_zip(fillvals, result))
      jnp_fun = lambda x: jnp.nonzero(x, size=size, fill_value=fill_value)
      with jtu.ignore_warning(category=DeprecationWarning,
                              message="Calling nonzero on 0d arrays.*"):
>       self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=False)

lax_stablehlo_reference_test.py:336: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1342: in _CheckAgainstNumpy
    lax_ans = lax_op(*args)
lax_stablehlo_reference_test.py:333: in <lambda>
    jnp_fun = lambda x: jnp.nonzero(x, size=size, fill_value=fill_value)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    def nonzero(a: ArrayLike, *, size: int | None = None,
                fill_value: None | ArrayLike | tuple[ArrayLike, ...] = None
        ) -> tuple[Array, ...]:
      """Return indices of nonzero elements of an array.
    
      JAX implementation of :func:`numpy.nonzero`.
    
      Because the size of the output of ``nonzero`` is data-dependent, the function
      is not compatible with JIT and other transformations. The JAX version adds
      the optional ``size`` argument which must be specified statically for
      ``jnp.nonzero`` to be used within JAX's transformations.
    
      Args:
        a: N-dimensional array.
        size: optional static integer specifying the number of nonzero entries to
          return. If there are more nonzero elements than the specified ``size``,
          then indices will be truncated at the end. If there are fewer nonzero
          elements than the specified size, then indices will be padded with
          ``fill_value``, which defaults to zero.
        fill_value: optional padding value when ``size`` is specified. Defaults to 0.
    
      Returns:
        Tuple of JAX Arrays of length ``a.ndim``, containing the indices of each
        nonzero value.
    
      See also:
        - :func:`jax.numpy.flatnonzero`
        - :func:`jax.numpy.where`
    
      Examples:
    
        One-dimensional array returns a length-1 tuple of indices:
    
        >>> x = jnp.array([0, 5, 0, 6, 0, 7])
        >>> jnp.nonzero(x)
        (Array([1, 3, 5], dtype=int32),)
    
        Two-dimensional array returns a length-2 tuple of indices:
    
        >>> x = jnp.array([[0, 5, 0],
        ...                [6, 0, 7]])
        >>> jnp.nonzero(x)
        (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))
    
        In either case, the resulting tuple of indices can be used directly to extract
        the nonzero values:
    
        >>> indices = jnp.nonzero(x)
        >>> x[indices]
        Array([5, 6, 7], dtype=int32)
    
        The output of ``nonzero`` has a dynamic shape, because the number of returned
        indices depends on the contents of the input array. As such, it is incompatible
        with JIT and other JAX transformations:
    
        >>> x = jnp.array([0, 5, 0, 6, 0, 7])
        >>> jax.jit(jnp.nonzero)(x)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
          ...
        ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[].
        The size argument of jnp.nonzero must be statically specified to use jnp.nonzero within JAX transformations.
    
        This can be addressed by passing a static ``size`` parameter to specify the
        desired output shape:
    
        >>> nonzero_jit = jax.jit(jnp.nonzero, static_argnames='size')
        >>> nonzero_jit(x, size=3)
        (Array([1, 3, 5], dtype=int32),)
    
        If ``size`` does not match the true size, the result will be either truncated or padded:
    
        >>> nonzero_jit(x, size=2)  # size < 3: indices are truncated
        (Array([1, 3], dtype=int32),)
        >>> nonzero_jit(x, size=5)  # size > 3: indices are padded with zeros.
        (Array([1, 3, 5, 0, 0], dtype=int32),)
    
        You can specify a custom fill value for the padding using the ``fill_value`` argument:
    
        >>> nonzero_jit(x, size=5, fill_value=len(x))
        (Array([1, 3, 5, 6, 6], dtype=int32),)
      """
      util.check_arraylike("nonzero", a)
      arr = asarray(a)
      del a
      if ndim(arr) == 0:
>       raise ValueError("Calling nonzero on 0d arrays is not allowed. "
                         "Use jnp.atleast_1d(scalar).nonzero() instead.")
E       ValueError: Calling nonzero on 0d arrays is not allowed. Use jnp.atleast_1d(scalar).nonzero() instead.

env/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py:3469: ValueError
_____________________________________________ LaxBackedNumpyTests.testTensordot5 ______________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testTensordot5>, lhs_shape = (1, 2, 3, 4)
lhs_dtype = <class 'ml_dtypes.bfloat16'>, rhs_shape = (4, 5, 3, 6), rhs_dtype = <class 'ml_dtypes.bfloat16'>
axes = [[2, 3], [2, 0]]

    @jtu.sample_product(
      [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, axes=axes)
        for lhs_shape, rhs_shape, axes in [
            [(3,), (), 0],
            [(2, 3, 4), (5, 6, 7), 0],  # from issue #740
            [(2, 3, 4), (3, 4, 5, 6), 2],
            [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
            [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
            [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
        ]],
      lhs_dtype=float_dtypes,#number_dtypes,
      rhs_dtype=float_dtypes,#number_dtypes,
    )
    @jax.default_matmul_precision("float32")
    def testTensordot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes):
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
      jnp_fun = lambda a, b: jnp.tensordot(a, b, axes)
      def np_fun(a, b):
        a = a if lhs_dtype != jnp.bfloat16 else a.astype(np.float32)
        b = b if rhs_dtype != jnp.bfloat16 else b.astype(np.float32)
        dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
        return np.tensordot(a, b, axes).astype(dtype)
      tol = {np.float16: 1e-1, np.float32: 1e-3, np.float64: 1e-12,
             np.complex64: 1e-3, np.complex128: 1e-12}
    
      with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
>       self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)

lax_stablehlo_reference_test.py:637: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1344: in _CheckAgainstNumpy
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f0290d1aca0>, array([[[[-22.125   ,  64.5     ,  58.25    ,  28.875   ,  46.5     ,
           73.5     ],
         [ 46.5     , -13.0625  , -79.5     ,  26.75    ,  51.      ,
          -59.5     ],
         [ 34.25    ,   6.625   ,  -8.5     , -35.75    ,  11.      ,
           35.5     ],
         [-21.625   , -15.125   ,  20.375   ,  47.      , -16.875   ,
           69.      ],
         [ 23.875   , -16.125   ,  90.      ,  -3.609375, -18.      ,
            9.125   ]],

        [[ 24.375   , -44.      , -37.      ,  -5.53125 ,  39.5     ,
           33.75    ],
         [ 51.5     , -65.      , -21.625   , -90.5     ,  27.375   ,
           82.5     ],
         [ -3.25    , -23.5     ,   8.0625  ,  10.      ,   9.75    ,
           66.      ],
         [ -4.75    ,  34.25    ,  27.875   , -76.      , -56.25    ,
          -37.      ],
         [-12.75    ,  69.      ,  32.25    ,   1.03125 , -11.0625  ,
           70.5     ]]]], dtype=float32), array([[[[-22.25   ,  64.5    ,  58.25   ,  28.625  ,  47.     ,
           73.5    ],
         [ 46.75   , -13.     , -80.     ,  26.75   ,  50.75   ,
          -59.5    ],
         [ 34.     ,   6.59375,  -8.5    , -36.25   ,  11.0625 ,
           35.75   ],
         [-21.625  , -15.1875 ,  20.5    ,  47.     , -17.125  ,
           68.5    ],
         [ 23.875  , -16.125  ,  90.     ,  -3.59375, -18.     ,
            9.125  ]],

        [[ 24.375  , -44.     , -37.     ,  -5.65625,  39.     ,
           33.75   ],
         [ 51.5    , -65.     , -22.5    , -90.5    ,  27.375  ,
           82.     ],
         [ -3.25   , -23.625  ,   8.1875 ,  10.0625 ,   9.8125 ,
           66.5    ],
         [ -4.625  ,  34.5    ,  28.25   , -76.     , -56.     ,
          -37.     ],
         [-12.625  ,  69.     ,  32.25   ,   1.0625 , -11.25   ,
           70.5    ]]]], dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.01, atol=0.01', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.01
E           
E           Mismatched elements: 11 / 60 (18.3%)
E           Max absolute difference among violations: 0.875
E           Max relative difference among violations: 0.03888889
E            ACTUAL: array([[[[-22.125   ,  64.5     ,  58.25    ,  28.875   ,  46.5     ,
E                      73.5     ],
E                    [ 46.5     , -13.0625  , -79.5     ,  26.75    ,  51.      ,...
E            DESIRED: array([[[[-22.25   ,  64.5    ,  58.25   ,  28.625  ,  47.     ,
E                      73.5    ],
E                    [ 46.75   , -13.     , -80.     ,  26.75   ,  50.75   ,...

/usr/lib/python3.11/contextlib.py:81: AssertionError
_____________________________________________ LaxBackedNumpyTests.testTensordot8 ______________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testTensordot8>, lhs_shape = (2, 3, 4)
lhs_dtype = <class 'ml_dtypes.bfloat16'>, rhs_shape = (5, 4, 3, 6), rhs_dtype = <class 'ml_dtypes.bfloat16'>, axes = [1, 2]

    @jtu.sample_product(
      [dict(lhs_shape=lhs_shape, rhs_shape=rhs_shape, axes=axes)
        for lhs_shape, rhs_shape, axes in [
            [(3,), (), 0],
            [(2, 3, 4), (5, 6, 7), 0],  # from issue #740
            [(2, 3, 4), (3, 4, 5, 6), 2],
            [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
            [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
            [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
        ]],
      lhs_dtype=float_dtypes,#number_dtypes,
      rhs_dtype=float_dtypes,#number_dtypes,
    )
    @jax.default_matmul_precision("float32")
    def testTensordot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes):
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
      jnp_fun = lambda a, b: jnp.tensordot(a, b, axes)
      def np_fun(a, b):
        a = a if lhs_dtype != jnp.bfloat16 else a.astype(np.float32)
        b = b if rhs_dtype != jnp.bfloat16 else b.astype(np.float32)
        dtype = jnp.promote_types(lhs_dtype, rhs_dtype)
        return np.tensordot(a, b, axes).astype(dtype)
      tol = {np.float16: 1e-1, np.float32: 1e-3, np.float64: 1e-12,
             np.complex64: 1e-3, np.complex128: 1e-12}
    
      with jtu.strict_promotion_if_dtypes_match([lhs_dtype, rhs_dtype]):
>       self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, tol=tol)

lax_stablehlo_reference_test.py:637: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1344: in _CheckAgainstNumpy
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f0290de8ea0>, array([[[[[-5.03125000e+00, -3.10937500e+00, -1.65000000e+01,
           -5.82031250e-01, -1.62500000e+00,  1.71875000e+00],
          [-1.55000000e+01,  7.25000000e+00, -6.59375000e+00,
            1.08750000e+01, -4.06250000e-01,  3.57812500e+00],
          [-9.50000000e+00,  2.03750000e+01,  1.50000000e+01,
            2.58750000e+01,  1.03125000e+01, -6.40625000e+00],
          [-9.68750000e+00, -7.09375000e+00, -2.40625000e+00,
           -8.37500000e+00,  1.04375000e+01,  2.31250000e+01]],

         [[ 1.13750000e+01, -1.02500000e+01,  1.05000000e+01,
           -3.98437500e+00, -1.89843750e+00,  4.78125000e+00],
          [ 5.90625000e+00,  5.37500000e+00, -6.00000000e+00,
            1.30000000e+01,  7.81250000e+00, -1.28125000e+01],
          [-8.62500000e+00, -1.55625000e+01,  9.25000000e+00,
           -1.36875000e+01, -3.59375000e+00,  9.13085938e-02],
          [ 1.62500000e+01,  5.78125000e+00,  1.86250000e+01,
            2.21250000e+01,  1.13125000e+01, -1.21875000e+01]],

         [[-5.71875000e+00,  1.17500000e+01,  1.66250000e+01,
            1.16250000e+01, -8.75000000e+00, -1.50000000e+01],
          [ 8.37500000e+00, -1.40625000e+01,  5.06250000e+00,
           -2.77343750e-01, -1.01250000e+01, -9.37500000e+00],
          [ 1.44531250e+00, -3.26562500e+00, -1.48437500e+00,
            1.30625000e+01,  3.64062500e+00, -5.23437500e-01],
          [ 1.55625000e+01, -8.24218750e-01,  3.30000000e+01,
            1.02500000e+01,  2.61250000e+01, -2.71875000e+00]],

         [[-6.53125000e+00, -5.78125000e+00,  1.88750000e+01,
           -3.43750000e-01,  3.82812500e-01, -2.21875000e+00],
          [-1.04375000e+01,  1.19375000e+01, -1.24375000e+01,
           -2.59375000e+00, -1.08125000e+01, -2.48750000e+01],
          [ 8.75000000e+00,  1.06875000e+01, -9.50000000e+00,
            1.81250000e+01, -1.86250000e+01,  2.07500000e+01],
          [-9.50000000e+00,  3.03955078e-02,  3.12500000e+00,
            2.51562500e+00, -7.28125000e+00,  1.68750000e+01]],

         [[ 2.37500000e+00,  1.13281250e+00, -5.65625000e+00,
            8.87500000e+00, -9.62500000e+00,  4.18750000e+00],
          [ 9.25000000e+00,  1.81640625e-01, -1.78750000e+01,
           -6.90625000e+00, -5.46875000e+00,  1.33750000e+01],
          [-1.05625000e+01,  8.00000000e+00, -4.06250000e+00,
           -6.75000000e+00, -4.84375000e+00,  7.21875000e+00],
          [ 5.87500000e+00, -3.17187500e+00,  2.35937500e+00,
           -1.66250000e+01, -9.68750000e+00,  1.35625000e+01]]],


        [[[-6.34375000e+00, -8.56250000e+00, -1.18125000e+01,
           -6.00000000e+00,  8.86718750e-01, -5.65625000e+00],
          [-1.15625000e+01,  6.56250000e+00, -5.18750000e+00,
            1.30000000e+01, -2.00000000e+00, -1.10156250e+00],
          [-4.12500000e+00,  7.68750000e+00,  1.42500000e+01,
            3.00000000e+01,  1.16875000e+01, -5.62500000e+00],
          [-3.81250000e+00, -6.28125000e+00,  2.07812500e+00,
           -3.32812500e+00,  6.25000000e+00,  1.25625000e+01]],

         [[ 9.06250000e+00, -1.13125000e+01,  1.19375000e+01,
           -6.75781250e-01,  9.08203125e-02, -6.31250000e+00],
          [ 1.19531250e+00,  7.75000000e+00,  1.35937500e+00,
            5.03125000e+00,  4.96875000e+00, -9.43750000e+00],
          [-6.32812500e-01, -7.71875000e+00,  1.01875000e+01,
           -1.52587891e-02, -5.84375000e+00, -7.43750000e+00],
          [ 1.71250000e+01,  5.40625000e+00,  1.26875000e+01,
            1.23125000e+01,  6.00000000e+00, -7.81250000e+00]],

         [[ 3.94531250e-01,  8.06250000e+00,  5.68750000e+00,
            2.73437500e+00, -1.08750000e+01, -7.43750000e+00],
          [ 1.10625000e+01, -1.08750000e+01, -8.20312500e-01,
            5.00000000e+00, -4.37500000e+00, -4.49218750e-02],
          [-9.25781250e-01,  1.71875000e+00,  4.74609375e-01,
            1.05000000e+01, -2.79296875e-01,  2.98437500e+00],
          [ 1.58750000e+01,  3.73046875e-01,  2.76250000e+01,
            9.62500000e+00,  2.07500000e+01, -8.06250000e+00]],

         [[-3.78125000e+00, -8.75000000e+00,  1.57500000e+01,
           -3.87500000e+00,  7.09375000e+00, -8.75000000e-01],
          [-4.53125000e+00,  7.84375000e+00,  2.42187500e+00,
            2.04687500e+00, -8.93750000e+00, -1.93750000e+01],
          [ 8.68750000e+00,  1.38750000e+01, -9.96093750e-01,
            8.31250000e+00, -3.46875000e+00,  1.62500000e+01],
          [-2.84375000e+00,  4.12500000e+00,  4.15625000e+00,
            2.93750000e+00, -1.01250000e+01,  1.24375000e+01]],

         [[-1.46875000e+00,  1.34375000e+00,  1.63281250e+00,
            7.81250000e+00, -1.38750000e+01, -3.35937500e+00],
          [ 3.07812500e+00, -8.75000000e+00, -1.05625000e+01,
           -2.25000000e+00,  3.47656250e-01,  1.56250000e+01],
          [-1.33125000e+01,  6.15625000e+00, -7.75000000e+00,
           -5.31250000e-01, -1.35000000e+01,  3.35937500e+00],
          [-1.89843750e+00, -6.78125000e+00,  1.20625000e+01,
           -1.86250000e+01, -3.78125000e+00,  1.63750000e+01]]],


        [[[-9.56250000e+00, -2.02500000e+01, -1.70000000e+01,
           -2.12500000e+01,  4.96875000e+00, -2.33750000e+01],
          [-1.31875000e+01,  9.12500000e+00, -5.34375000e+00,
            2.68750000e+01, -6.96875000e+00, -9.37500000e+00],
          [ 9.64843750e-01, -8.31250000e+00,  2.62500000e+01,
            6.37500000e+01,  2.21250000e+01, -7.28125000e+00],
          [ 2.90625000e+00, -9.87500000e+00,  1.23125000e+01,
            2.25000000e+00,  3.54687500e+00,  1.27343750e+00]],

         [[ 5.65625000e+00, -2.30000000e+01,  2.72500000e+01,
            2.85937500e+00,  4.53125000e+00, -3.05000000e+01],
          [-6.12500000e+00,  1.95000000e+01,  1.87500000e+01,
           -5.15625000e+00,  1.52343750e+00, -7.09375000e+00],
          [ 1.15000000e+01,  7.53125000e+00,  1.93750000e+01,
            3.37500000e+01, -1.63750000e+01, -3.11250000e+01],
          [ 3.77500000e+01,  7.06250000e+00,  1.78750000e+01,
           -1.18750000e+00,  7.06250000e+00, -2.46093750e-01]],

         [[ 1.38750000e+01,  6.34375000e+00, -7.71875000e+00,
           -9.87500000e+00, -2.41250000e+01,  2.12500000e+00],
          [ 2.38750000e+01, -1.41875000e+01, -1.95000000e+01,
            2.20000000e+01,  2.21875000e+00,  1.77500000e+01],
          [-6.15625000e+00,  1.28750000e+01,  7.85156250e-01,
            1.61250000e+01, -2.78125000e+00,  1.39375000e+01],
          [ 2.82500000e+01,  2.98828125e-01,  3.25000000e+01,
            1.33750000e+01,  2.45000000e+01, -2.22500000e+01]],

         [[-2.95312500e+00, -2.18750000e+01,  1.55000000e+01,
           -1.33750000e+01,  3.12500000e+01,  2.57812500e+00],
          [ 5.03125000e+00,  8.62500000e+00,  3.25000000e+01,
            1.23125000e+01, -9.62500000e+00, -1.95000000e+01],
          [ 1.93750000e+01,  3.01250000e+01,  1.63750000e+01,
           -1.67968750e+00,  2.81250000e+01,  2.30000000e+01],
          [ 1.15000000e+01,  1.15000000e+01,  6.62500000e+00,
            6.28125000e+00, -1.87500000e+01,  1.22500000e+01]],

         [[-1.17500000e+01, -1.89062500e+00,  1.77500000e+01,
            8.31250000e+00, -2.88750000e+01, -2.07500000e+01],
          [-9.00000000e+00, -3.72500000e+01, -5.15625000e-01,
            4.40625000e+00,  7.87500000e+00,  3.45000000e+01],
          [-3.47500000e+01,  2.48437500e+00, -2.03750000e+01,
            1.04375000e+01, -4.25000000e+01,  1.26562500e+00],
          [-1.95000000e+01, -1.97500000e+01,  3.87500000e+01,
           -3.80000000e+01,  2.35937500e+00,  3.02500000e+01]]],


        [[[ 9.18750000e+00,  1.63750000e+01,  1.36250000e+01,
            1.45625000e+01, -3.53125000e+00,  1.58750000e+01],
          [ 1.25000000e+01, -8.31250000e+00,  5.59375000e+00,
           -2.10000000e+01,  4.75000000e+00,  6.15625000e+00],
          [ 7.77343750e-01,  1.88281250e+00, -2.05000000e+01,
           -4.85000000e+01, -1.78750000e+01,  6.84375000e+00],
          [-3.39843750e-01,  8.31250000e+00, -7.87500000e+00,
           -1.54296875e-01, -4.65625000e+00, -6.40625000e+00]],

         [[-8.56250000e+00,  1.77500000e+01, -1.98750000e+01,
           -1.84375000e+00, -2.46875000e+00,  2.05000000e+01],
          [ 3.12500000e+00, -1.41250000e+01, -1.06875000e+01,
            9.88281250e-01, -3.45312500e+00,  8.87500000e+00],
          [-6.87500000e+00,  5.31250000e-01, -1.55625000e+01,
           -1.73750000e+01,  1.14375000e+01,  2.00000000e+01],
          [-2.76250000e+01, -6.78125000e+00, -1.41250000e+01,
           -5.68750000e+00, -5.12500000e+00,  4.75000000e+00]],

         [[-7.71875000e+00, -7.34375000e+00,  2.59375000e+00,
            4.93750000e+00,  1.82500000e+01,  2.23437500e+00],
          [-1.85000000e+01,  1.26250000e+01,  9.81250000e+00,
           -1.40000000e+01,  4.19921875e-01, -1.00625000e+01],
          [ 3.89062500e+00, -8.00000000e+00, -1.53125000e+00,
           -1.31250000e+01,  2.89062500e+00, -8.75000000e+00],
          [-2.30000000e+01, -9.41406250e-01, -3.20000000e+01,
           -1.23750000e+01, -2.35000000e+01,  1.66250000e+01]],

         [[ 2.98437500e+00,  1.61250000e+01, -1.72500000e+01,
            9.31250000e+00, -1.95000000e+01, -7.18750000e-01],
          [-4.41406250e-01, -7.71875000e+00, -1.95000000e+01,
           -7.93750000e+00,  1.00000000e+01,  2.06250000e+01],
          [-1.38125000e+01, -2.32500000e+01, -8.43750000e+00,
           -2.10937500e+00, -1.31875000e+01, -1.93750000e+01],
          [-4.37500000e+00, -9.12500000e+00, -6.21875000e+00,
           -4.78125000e+00,  1.61250000e+01, -1.26875000e+01]],

         [[ 6.90625000e+00, -5.97656250e-01, -1.05625000e+01,
           -8.93750000e+00,  2.32500000e+01,  1.31875000e+01],
          [ 3.14062500e+00,  2.37500000e+01,  5.84375000e+00,
           -1.63281250e+00, -5.50000000e+00, -2.58750000e+01],
          [ 2.41250000e+01, -5.25000000e+00,  1.50625000e+01,
           -5.81250000e+00,  2.96250000e+01, -1.57031250e+00],
          [ 1.15625000e+01,  1.39375000e+01, -2.75000000e+01,
            2.95000000e+01, -1.86523438e-01, -2.52500000e+01]]]],



       [[[[ 2.79687500e+00,  7.15625000e+00, -5.37500000e+00,
            5.50000000e+00, -2.87500000e+00,  7.65625000e+00],
          [-3.31250000e+00, -3.49121094e-02, -7.30468750e-01,
           -2.81250000e+00,  1.55468750e+00,  5.03125000e+00],
          [-5.59375000e+00,  1.25000000e+01,  6.67968750e-01,
           -4.71875000e+00, -2.25000000e+00, -1.55273438e-01],
          [-5.90625000e+00, -5.07812500e-01, -4.46875000e+00,
           -5.12500000e+00,  3.85937500e+00,  9.37500000e+00]],

         [[ 1.25732422e-02,  1.44531250e+00, -1.12500000e+00,
           -3.81250000e+00, -1.76562500e+00,  1.15625000e+01],
          [ 4.71875000e+00, -2.37500000e+00, -6.56250000e+00,
            7.75000000e+00,  1.98437500e+00, -1.85937500e+00],
          [-8.50000000e+00, -5.50000000e+00, -1.58593750e+00,
           -1.11250000e+01,  1.99218750e+00,  6.68750000e+00],
          [-1.74804688e-01, -4.25781250e-01,  7.00000000e+00,
            7.65625000e+00,  6.62500000e+00, -2.39062500e+00]],

         [[-5.53125000e+00,  2.87500000e+00,  1.11250000e+01,
            9.12500000e+00,  2.45312500e+00, -6.71875000e+00],
          [-3.42187500e+00, -2.70312500e+00,  3.85937500e+00,
           -4.53125000e+00, -5.62500000e+00, -9.06250000e+00],
          [ 2.35937500e+00, -4.84375000e+00, -2.98437500e+00,
            2.67187500e+00,  5.28125000e+00, -3.03125000e+00],
          [-1.22656250e+00, -1.92187500e+00,  1.75781250e+00,
           -6.48437500e-01,  3.54687500e+00,  6.18750000e+00]],

         [[-2.87500000e+00,  3.15625000e+00,  2.87109375e-01,
            3.68750000e+00, -5.28125000e+00, -8.20312500e-01],
          [-4.93750000e+00,  4.28125000e+00, -1.46250000e+01,
           -4.65625000e+00, -5.39062500e-01, -2.93750000e+00],
          [ 7.18750000e-01, -3.93750000e+00, -7.68750000e+00,
            9.81250000e+00, -1.30000000e+01,  4.40625000e+00],
          [-5.21875000e+00, -5.18750000e+00, -1.98437500e+00,
           -4.94140625e-01,  4.59375000e+00,  3.29687500e+00]],

         [[ 3.29687500e+00, -1.58593750e+00, -6.87500000e+00,
           -3.47656250e-01,  5.90625000e+00,  7.37500000e+00],
          [ 5.03125000e+00,  7.78125000e+00, -5.43750000e+00,
           -4.53125000e+00, -6.75000000e+00, -2.34375000e+00],
          [ 1.71875000e+00,  8.78906250e-02,  4.12500000e+00,
           -6.15625000e+00,  8.62500000e+00,  4.31250000e+00],
          [ 7.31250000e+00,  3.65625000e+00, -1.03750000e+01,
            2.79687500e+00, -6.12500000e+00, -4.81250000e+00]]],


        [[[ 7.84375000e+00,  1.05000000e+01, -3.11250000e+01,
           -5.18750000e+00, -4.62500000e+00, -7.14843750e-01],
          [-1.47500000e+01,  3.15625000e+00, -1.96875000e+00,
            1.23125000e+01, -2.76562500e+00,  5.65625000e+00],
          [-1.32500000e+01,  2.05000000e+01,  2.36250000e+01,
            3.97500000e+01,  8.18750000e+00, -2.39062500e+00],
          [-1.13750000e+01, -7.81250000e+00,  6.17187500e-01,
           -1.03750000e+01,  9.68750000e+00,  1.45000000e+01]],

         [[-9.87500000e+00, -1.41875000e+01,  2.30000000e+01,
           -9.50000000e+00,  1.26562500e+00,  3.70312500e+00],
          [ 5.46875000e+00,  1.13125000e+01,  6.34375000e+00,
            1.20000000e+01,  2.25585938e-01, -4.94140625e-01],
          [-1.25625000e+01,  9.50000000e+00,  8.93750000e+00,
            2.05000000e+01, -1.11250000e+01, -1.68750000e+01],
          [ 3.65000000e+01, -1.48437500e-01,  3.80000000e+01,
            2.50000000e+00,  3.00000000e+01,  7.43750000e+00]],

         [[ 3.09375000e+00,  6.81250000e+00,  2.01250000e+01,
            1.38750000e+01, -1.31250000e+01, -7.96875000e+00],
          [ 7.78125000e+00, -1.55625000e+01, -2.11250000e+01,
            1.36250000e+01, -1.01875000e+01, -3.64062500e+00],
          [-2.86865234e-02,  1.03125000e+00, -1.29375000e+01,
            2.05000000e+01,  1.85000000e+01,  8.25000000e+00],
          [ 1.55000000e+01, -8.87500000e+00,  9.37500000e+00,
            1.87500000e+00,  1.71250000e+01,  8.47656250e-01]],

         [[-1.00000000e+01, -1.05625000e+01, -4.21875000e+00,
           -2.09375000e+00,  2.41250000e+01,  3.79687500e+00],
          [-7.65625000e-01,  1.83750000e+01, -3.76562500e+00,
           -1.02539062e-01, -1.00000000e+00, -7.25000000e+00],
          [ 2.26250000e+01,  1.18750000e+01,  2.10937500e+00,
            2.06250000e+01,  8.56250000e+00,  2.87500000e+01],
          [ 7.43750000e+00, -9.18750000e+00, -5.06250000e+00,
            3.82812500e+00,  5.84375000e+00,  1.05625000e+01]],

         [[-6.18750000e+00, -1.42500000e+01,  2.18750000e+00,
           -2.92187500e+00, -3.65234375e-01, -2.10937500e+00],
          [-3.56250000e+00, -2.15000000e+01, -3.00781250e-01,
           -5.75000000e+00, -1.47500000e+01,  2.41250000e+01],
          [-3.25000000e+01, -9.18750000e+00, -5.40625000e+00,
           -4.71875000e+00, -1.76250000e+01,  1.39375000e+01],
          [-3.14062500e+00, -8.37500000e+00,  5.43750000e+00,
           -2.16250000e+01, -1.35625000e+01,  2.28125000e+00]]],


        [[[ 2.20000000e+01,  3.90000000e+01, -5.65625000e+00,
            2.37500000e+01, -1.12500000e+01,  3.08750000e+01],
          [ 5.78125000e+00, -1.05625000e+01,  6.50000000e+00,
           -2.61250000e+01,  6.78125000e+00,  1.80000000e+01],
          [-1.31250000e+01,  2.86250000e+01, -1.44375000e+01,
           -5.10000000e+01, -2.38750000e+01,  8.62500000e+00],
          [-1.38750000e+01,  6.78125000e+00, -1.56875000e+01,
           -1.21250000e+01,  3.15625000e+00,  8.81250000e+00]],

         [[-2.00000000e+01,  1.87500000e+01, -1.55625000e+01,
           -1.29375000e+01, -4.53125000e+00,  4.45000000e+01],
          [ 1.28750000e+01, -1.60000000e+01, -1.80000000e+01,
            1.70000000e+01, -3.28125000e+00,  1.15625000e+01],
          [-2.70000000e+01,  2.31250000e+00, -1.91250000e+01,
           -2.27500000e+01,  1.15625000e+01,  2.51250000e+01],
          [-1.70000000e+01, -1.08750000e+01,  1.11875000e+01,
            1.20117188e-01,  1.90000000e+01,  1.01875000e+01]],

         [[-1.47500000e+01, -3.85937500e+00,  2.81250000e+01,
            2.56250000e+01,  2.10000000e+01, -8.31250000e+00],
          [-2.61250000e+01,  6.00000000e+00,  3.95312500e+00,
           -1.61250000e+01, -1.15625000e+01, -2.62500000e+01],
          [ 8.12500000e+00, -1.59375000e+01, -1.40000000e+01,
           -3.45312500e+00,  2.21250000e+01, -1.04375000e+01],
          [-2.56250000e+01, -9.37500000e+00, -4.10000000e+01,
           -1.82500000e+01, -2.06250000e+01,  3.16250000e+01]],

         [[-5.00000000e+00,  2.02500000e+01, -2.88750000e+01,
            1.62500000e+01, -1.78750000e+01,  7.69531250e-01],
          [-5.75000000e+00,  4.87500000e+00, -4.60000000e+01,
           -1.65000000e+01,  1.40625000e+01,  2.37500000e+01],
          [-4.75000000e+00, -3.08750000e+01, -1.85000000e+01,
            2.01250000e+01, -2.62500000e+01, -5.68750000e+00],
          [-6.37500000e+00, -2.51250000e+01, -1.48125000e+01,
           -5.12500000e+00,  3.30000000e+01, -9.06250000e+00]],

         [[ 9.31250000e+00, -1.22500000e+01, -2.10000000e+01,
           -1.60000000e+01,  4.07500000e+01,  2.55000000e+01],
          [ 6.96875000e+00,  2.86250000e+01,  3.68750000e+00,
           -1.06250000e+01, -2.48750000e+01, -2.50000000e+01],
          [ 1.58750000e+01, -1.43125000e+01,  2.30000000e+01,
           -1.78750000e+01,  4.10000000e+01,  1.11875000e+01],
          [ 2.22500000e+01,  1.88750000e+01, -4.77500000e+01,
            3.27500000e+01, -1.53125000e+01, -4.15000000e+01]]],


        [[[ 9.25000000e+00,  1.62500000e+01, -3.00000000e+01,
            2.25000000e+00, -6.81250000e+00,  8.12500000e+00],
          [-1.50000000e+01,  2.34375000e+00, -2.29687500e+00,
            6.18750000e+00, -3.59375000e-01,  1.00625000e+01],
          [-1.65000000e+01,  2.98750000e+01,  1.88750000e+01,
            2.50000000e+01,  3.65625000e+00, -1.96875000e+00],
          [-1.54375000e+01, -6.53125000e+00, -4.56250000e+00,
           -1.37500000e+01,  1.17500000e+01,  2.16250000e+01]],

         [[-7.68750000e+00, -9.18750000e+00,  1.63750000e+01,
           -1.16250000e+01, -1.01562500e+00,  1.60000000e+01],
          [ 9.50000000e+00,  5.96875000e+00, -2.53125000e+00,
            1.80000000e+01,  2.37500000e+00, -2.39062500e+00],
          [-1.92500000e+01,  1.20312500e+00,  5.00000000e+00,
            3.20312500e+00, -6.28125000e+00, -5.37500000e+00],
          [ 2.78750000e+01, -6.44531250e-01,  3.70000000e+01,
            1.04375000e+01,  3.05000000e+01,  3.10937500e+00]],

         [[-3.85937500e+00,  8.43750000e+00,  2.80000000e+01,
            2.10000000e+01, -7.25000000e+00, -1.36875000e+01],
          [ 2.04687500e+00, -1.49375000e+01, -1.19375000e+01,
            5.34375000e+00, -1.41250000e+01, -1.30000000e+01],
          [ 2.64062500e+00, -4.68750000e+00, -1.33125000e+01,
            1.87500000e+01,  2.02500000e+01,  2.93750000e+00],
          [ 1.04375000e+01, -9.00000000e+00,  8.93750000e+00,
            6.25000000e-01,  1.70000000e+01,  7.71875000e+00]],

         [[-1.09375000e+01, -4.46875000e+00, -3.06250000e+00,
            2.57812500e+00,  1.25625000e+01,  2.01562500e+00],
          [-6.12500000e+00,  1.90000000e+01, -1.93750000e+01,
           -5.34375000e+00, -1.29687500e+00, -8.75000000e+00],
          [ 1.82500000e+01,  4.62500000e+00, -7.03125000e+00,
            2.68750000e+01, -8.00000000e+00,  2.70000000e+01],
          [-1.15722656e-01, -1.30000000e+01, -6.18750000e+00,
            2.37500000e+00,  9.81250000e+00,  1.17500000e+01]],

         [[-1.03906250e+00, -1.28125000e+01, -6.09375000e+00,
           -2.71875000e+00,  6.50000000e+00,  6.75000000e+00],
          [ 2.90625000e+00, -7.71875000e+00, -6.28125000e+00,
           -9.50000000e+00, -1.90000000e+01,  1.57500000e+01],
          [-2.30000000e+01, -7.03125000e+00,  5.35156250e-01,
           -1.06250000e+01, -3.79687500e+00,  1.55625000e+01],
          [ 5.84375000e+00, -2.26562500e+00, -7.56250000e+00,
           -1.33750000e+01, -1.72500000e+01, -3.79687500e+00]]]]],
      dtype=float32), array([[[[[-5.0000000e+00, -3.1250000e+00, -1.6500000e+01,
           -5.6250000e-01, -1.6093750e+00,  1.6875000e+00],
          [-1.5500000e+01,  7.2500000e+00, -6.5937500e+00,
            1.0875000e+01, -3.9843750e-01,  3.5625000e+00],
          [-9.5625000e+00,  2.0375000e+01,  1.5000000e+01,
            2.5875000e+01,  1.0375000e+01, -6.4375000e+00],
          [-9.6875000e+00, -7.0625000e+00, -2.4218750e+00,
           -8.3750000e+00,  1.0437500e+01,  2.3125000e+01]],

         [[ 1.1375000e+01, -1.0250000e+01,  1.0500000e+01,
           -3.9687500e+00, -1.8984375e+00,  4.8125000e+00],
          [ 5.9062500e+00,  5.3750000e+00, -6.0312500e+00,
            1.3000000e+01,  7.8125000e+00, -1.2812500e+01],
          [-8.6250000e+00, -1.5562500e+01,  9.2500000e+00,
           -1.3750000e+01, -3.5937500e+00,  9.3750000e-02],
          [ 1.6250000e+01,  5.8125000e+00,  1.8625000e+01,
            2.2000000e+01,  1.1312500e+01, -1.2187500e+01]],

         [[-5.7500000e+00,  1.1750000e+01,  1.6500000e+01,
            1.1625000e+01, -8.7500000e+00, -1.5000000e+01],
          [ 8.3750000e+00, -1.4000000e+01,  5.0625000e+00,
           -2.7343750e-01, -1.0125000e+01, -9.3750000e+00],
          [ 1.4531250e+00, -3.2656250e+00, -1.4687500e+00,
            1.3062500e+01,  3.6562500e+00, -5.2343750e-01],
          [ 1.5562500e+01, -8.1250000e-01,  3.3000000e+01,
            1.0187500e+01,  2.6125000e+01, -2.7187500e+00]],

         [[-6.5000000e+00, -5.7812500e+00,  1.8875000e+01,
           -3.4375000e-01,  3.8671875e-01, -2.2187500e+00],
          [-1.0437500e+01,  1.1937500e+01, -1.2437500e+01,
           -2.5937500e+00, -1.0750000e+01, -2.4875000e+01],
          [ 8.6875000e+00,  1.0750000e+01, -9.5625000e+00,
            1.8250000e+01, -1.8625000e+01,  2.0750000e+01],
          [-9.5000000e+00,  6.2500000e-02,  3.1406250e+00,
            2.5156250e+00, -7.2812500e+00,  1.6875000e+01]],

         [[ 2.3750000e+00,  1.1562500e+00, -5.6562500e+00,
            8.8750000e+00, -9.6250000e+00,  4.2187500e+00],
          [ 9.2500000e+00,  1.8750000e-01, -1.8000000e+01,
           -6.9062500e+00, -5.4687500e+00,  1.3375000e+01],
          [-1.0625000e+01,  8.0000000e+00, -4.0625000e+00,
           -6.8125000e+00, -4.8125000e+00,  7.2187500e+00],
          [ 5.8750000e+00, -3.1562500e+00,  2.3750000e+00,
           -1.6625000e+01, -9.6875000e+00,  1.3500000e+01]]],


        [[[-6.3125000e+00, -8.5625000e+00, -1.1812500e+01,
           -5.9687500e+00,  8.7500000e-01, -5.6250000e+00],
          [-1.1562500e+01,  6.5625000e+00, -5.2187500e+00,
            1.3000000e+01, -2.0000000e+00, -1.0937500e+00],
          [-4.1250000e+00,  7.7187500e+00,  1.4250000e+01,
            3.0000000e+01,  1.1625000e+01, -5.5937500e+00],
          [-3.7968750e+00, -6.2812500e+00,  2.0781250e+00,
           -3.3437500e+00,  6.2500000e+00,  1.2625000e+01]],

         [[ 9.1250000e+00, -1.1375000e+01,  1.1937500e+01,
           -6.7187500e-01,  9.0332031e-02, -6.3125000e+00],
          [ 1.1953125e+00,  7.7187500e+00,  1.3437500e+00,
            5.0000000e+00,  4.9687500e+00, -9.5000000e+00],
          [-6.5625000e-01, -7.7500000e+00,  1.0187500e+01,
            0.0000000e+00, -5.8125000e+00, -7.4375000e+00],
          [ 1.7250000e+01,  5.4375000e+00,  1.2750000e+01,
            1.2312500e+01,  6.0312500e+00, -7.8125000e+00]],

         [[ 3.9453125e-01,  8.0625000e+00,  5.6875000e+00,
            2.7343750e+00, -1.0875000e+01, -7.4375000e+00],
          [ 1.1062500e+01, -1.0875000e+01, -8.4375000e-01,
            5.0000000e+00, -4.3750000e+00, -3.1250000e-02],
          [-9.2968750e-01,  1.7187500e+00,  4.3750000e-01,
            1.0500000e+01, -2.8125000e-01,  3.0000000e+00],
          [ 1.5875000e+01,  3.5937500e-01,  2.7625000e+01,
            9.6250000e+00,  2.0875000e+01, -8.0625000e+00]],

         [[-3.7812500e+00, -8.7500000e+00,  1.5750000e+01,
           -3.8750000e+00,  7.0937500e+00, -8.7500000e-01],
          [-4.5312500e+00,  7.8437500e+00,  2.4375000e+00,
            2.0468750e+00, -8.9375000e+00, -1.9375000e+01],
          [ 8.6875000e+00,  1.3875000e+01, -1.0000000e+00,
            8.3125000e+00, -3.4687500e+00,  1.6250000e+01],
          [-2.8437500e+00,  4.1250000e+00,  4.1562500e+00,
            2.9375000e+00, -1.0125000e+01,  1.2375000e+01]],

         [[-1.4687500e+00,  1.3437500e+00,  1.6328125e+00,
            7.8125000e+00, -1.3812500e+01, -3.3750000e+00],
          [ 3.0937500e+00, -8.7500000e+00, -1.0500000e+01,
           -2.2500000e+00,  3.4375000e-01,  1.5625000e+01],
          [-1.3250000e+01,  6.1562500e+00, -7.7500000e+00,
           -5.3125000e-01, -1.3500000e+01,  3.3437500e+00],
          [-1.9218750e+00, -6.8125000e+00,  1.2000000e+01,
           -1.8750000e+01, -3.7656250e+00,  1.6375000e+01]]],


        [[[-9.5000000e+00, -2.0250000e+01, -1.7000000e+01,
           -2.1250000e+01,  5.0000000e+00, -2.3250000e+01],
          [-1.3187500e+01,  9.1250000e+00, -5.3750000e+00,
            2.6875000e+01, -6.9687500e+00, -9.3750000e+00],
          [ 9.5312500e-01, -8.3125000e+00,  2.6250000e+01,
            6.3750000e+01,  2.2125000e+01, -7.2500000e+00],
          [ 2.8906250e+00, -9.8750000e+00,  1.2312500e+01,
            2.2343750e+00,  3.5625000e+00,  1.1875000e+00]],

         [[ 5.5000000e+00, -2.3000000e+01,  2.7250000e+01,
            2.8750000e+00,  4.5312500e+00, -3.0500000e+01],
          [-6.1250000e+00,  1.9500000e+01,  1.8750000e+01,
           -5.1562500e+00,  1.5312500e+00, -7.1250000e+00],
          [ 1.1500000e+01,  7.5000000e+00,  1.9375000e+01,
            3.4000000e+01, -1.6375000e+01, -3.1125000e+01],
          [ 3.7750000e+01,  7.0937500e+00,  1.7875000e+01,
           -1.2500000e+00,  7.0000000e+00, -2.5000000e-01]],

         [[ 1.3875000e+01,  6.3750000e+00, -7.7500000e+00,
           -9.8750000e+00, -2.4000000e+01,  2.1562500e+00],
          [ 2.3875000e+01, -1.4187500e+01, -1.9500000e+01,
            2.2000000e+01,  2.2187500e+00,  1.7750000e+01],
          [-6.1250000e+00,  1.2875000e+01,  7.5000000e-01,
            1.6125000e+01, -2.7500000e+00,  1.3937500e+01],
          [ 2.8375000e+01,  2.8125000e-01,  3.2500000e+01,
            1.3375000e+01,  2.4500000e+01, -2.2250000e+01]],

         [[-2.9687500e+00, -2.2000000e+01,  1.5500000e+01,
           -1.3375000e+01,  3.1250000e+01,  2.5781250e+00],
          [ 5.0312500e+00,  8.6250000e+00,  3.2500000e+01,
            1.2312500e+01, -9.6250000e+00, -1.9500000e+01],
          [ 1.9250000e+01,  3.0125000e+01,  1.6500000e+01,
           -1.7109375e+00,  2.8125000e+01,  2.3000000e+01],
          [ 1.1437500e+01,  1.1500000e+01,  6.6250000e+00,
            6.2812500e+00, -1.8750000e+01,  1.2250000e+01]],

         [[-1.1750000e+01, -1.8125000e+00,  1.7750000e+01,
            8.2500000e+00, -2.8875000e+01, -2.0750000e+01],
          [-9.0000000e+00, -3.7250000e+01, -6.2500000e-01,
            4.4062500e+00,  7.8437500e+00,  3.4500000e+01],
          [-3.4500000e+01,  2.5000000e+00, -2.0375000e+01,
            1.0375000e+01, -4.2500000e+01,  1.2656250e+00],
          [-1.9500000e+01, -1.9750000e+01,  3.8750000e+01,
           -3.8000000e+01,  2.3593750e+00,  3.0250000e+01]]],


        [[[ 9.1250000e+00,  1.6375000e+01,  1.3625000e+01,
            1.4500000e+01, -3.5312500e+00,  1.5875000e+01],
          [ 1.2437500e+01, -8.3125000e+00,  5.5625000e+00,
           -2.1000000e+01,  4.7500000e+00,  6.1250000e+00],
          [ 7.7343750e-01,  1.8906250e+00, -2.0500000e+01,
           -4.8500000e+01, -1.8000000e+01,  6.8750000e+00],
          [-3.3593750e-01,  8.3125000e+00, -7.8750000e+00,
           -1.4843750e-01, -4.6562500e+00, -6.4062500e+00]],

         [[-8.5625000e+00,  1.7750000e+01, -2.0000000e+01,
           -1.8593750e+00, -2.4843750e+00,  2.0500000e+01],
          [ 3.1250000e+00, -1.4125000e+01, -1.0687500e+01,
            9.8828125e-01, -3.4375000e+00,  8.8750000e+00],
          [-6.8750000e+00,  5.0000000e-01, -1.5562500e+01,
           -1.7375000e+01,  1.1437500e+01,  2.0000000e+01],
          [-2.7750000e+01, -6.7812500e+00, -1.4125000e+01,
           -5.6250000e+00, -5.1562500e+00,  4.7500000e+00]],

         [[-7.7187500e+00, -7.3125000e+00,  2.5937500e+00,
            4.9375000e+00,  1.8250000e+01,  2.2187500e+00],
          [-1.8500000e+01,  1.2625000e+01,  9.8125000e+00,
           -1.3937500e+01,  4.1992188e-01, -1.0000000e+01],
          [ 3.8906250e+00, -8.0000000e+00, -1.5625000e+00,
           -1.3125000e+01,  2.9375000e+00, -8.7500000e+00],
          [-2.3000000e+01, -9.3750000e-01, -3.2000000e+01,
           -1.2375000e+01, -2.3500000e+01,  1.6625000e+01]],

         [[ 2.9687500e+00,  1.6125000e+01, -1.7250000e+01,
            9.3125000e+00, -1.9500000e+01, -7.1875000e-01],
          [-4.3750000e-01, -7.7500000e+00, -1.9500000e+01,
           -7.9062500e+00,  1.0000000e+01,  2.0500000e+01],
          [-1.3875000e+01, -2.3250000e+01, -8.4375000e+00,
           -2.1250000e+00, -1.3250000e+01, -1.9500000e+01],
          [-4.3750000e+00, -9.1250000e+00, -6.1875000e+00,
           -4.7812500e+00,  1.6250000e+01, -1.2687500e+01]],

         [[ 6.9062500e+00, -6.2500000e-01, -1.0562500e+01,
           -8.9375000e+00,  2.3250000e+01,  1.3125000e+01],
          [ 3.0937500e+00,  2.3750000e+01,  5.8125000e+00,
           -1.6328125e+00, -5.5000000e+00, -2.5875000e+01],
          [ 2.4250000e+01, -5.2500000e+00,  1.5000000e+01,
           -5.8125000e+00,  2.9750000e+01, -1.5312500e+00],
          [ 1.1562500e+01,  1.3937500e+01, -2.7500000e+01,
            2.9500000e+01, -1.8750000e-01, -2.5250000e+01]]]],



       [[[[ 2.7968750e+00,  7.1875000e+00, -5.3125000e+00,
            5.5000000e+00, -2.8593750e+00,  7.6250000e+00],
          [-3.3125000e+00, -3.9062500e-02, -7.3046875e-01,
           -2.8125000e+00,  1.5546875e+00,  5.0000000e+00],
          [-5.5937500e+00,  1.2500000e+01,  6.7968750e-01,
           -4.7187500e+00, -2.2500000e+00, -1.6015625e-01],
          [-5.9062500e+00, -5.0390625e-01, -4.4687500e+00,
           -5.1250000e+00,  3.8437500e+00,  9.3750000e+00]],

         [[ 1.5625000e-02,  1.4375000e+00, -1.1328125e+00,
           -3.7968750e+00, -1.7656250e+00,  1.1562500e+01],
          [ 4.7187500e+00, -2.3750000e+00, -6.5625000e+00,
            7.7500000e+00,  1.9843750e+00, -1.8671875e+00],
          [-8.5000000e+00, -5.5000000e+00, -1.5781250e+00,
           -1.1125000e+01,  2.0000000e+00,  6.6875000e+00],
          [-1.8750000e-01, -4.2187500e-01,  6.9687500e+00,
            7.6250000e+00,  6.6250000e+00, -2.3750000e+00]],

         [[-5.5312500e+00,  2.8750000e+00,  1.1125000e+01,
            9.1250000e+00,  2.4531250e+00, -6.7187500e+00],
          [-3.4218750e+00, -2.6875000e+00,  3.8906250e+00,
           -4.5312500e+00, -5.5937500e+00, -9.0625000e+00],
          [ 2.3437500e+00, -4.8437500e+00, -2.9687500e+00,
            2.6718750e+00,  5.2500000e+00, -3.0312500e+00],
          [-1.2187500e+00, -1.9218750e+00,  1.7187500e+00,
           -6.5625000e-01,  3.5468750e+00,  6.1875000e+00]],

         [[-2.8906250e+00,  3.1562500e+00,  2.8125000e-01,
            3.6875000e+00, -5.2812500e+00, -8.1640625e-01],
          [-4.9375000e+00,  4.2812500e+00, -1.4625000e+01,
           -4.6562500e+00, -5.3906250e-01, -2.9375000e+00],
          [ 7.0703125e-01, -3.9218750e+00, -7.6875000e+00,
            9.8125000e+00, -1.2937500e+01,  4.4062500e+00],
          [-5.1875000e+00, -5.1875000e+00, -1.9765625e+00,
           -4.9218750e-01,  4.5937500e+00,  3.2968750e+00]],

         [[ 3.2968750e+00, -1.5937500e+00, -6.8750000e+00,
           -3.4375000e-01,  5.8750000e+00,  7.3750000e+00],
          [ 5.0312500e+00,  7.8125000e+00, -5.4687500e+00,
           -4.5312500e+00, -6.7500000e+00, -2.3437500e+00],
          [ 1.7031250e+00,  8.5937500e-02,  4.1250000e+00,
           -6.1875000e+00,  8.6250000e+00,  4.3125000e+00],
          [ 7.2812500e+00,  3.6562500e+00, -1.0375000e+01,
            2.7968750e+00, -6.1250000e+00, -4.8125000e+00]]],


        [[[ 7.8437500e+00,  1.0500000e+01, -3.1250000e+01,
           -5.1875000e+00, -4.5937500e+00, -7.1875000e-01],
          [-1.4750000e+01,  3.1718750e+00, -1.9765625e+00,
            1.2312500e+01, -2.7656250e+00,  5.6250000e+00],
          [-1.3250000e+01,  2.0500000e+01,  2.3625000e+01,
            3.9750000e+01,  8.1875000e+00, -2.4062500e+00],
          [-1.1375000e+01, -7.8437500e+00,  6.4062500e-01,
           -1.0375000e+01,  9.6875000e+00,  1.4437500e+01]],

         [[-9.9375000e+00, -1.4125000e+01,  2.3000000e+01,
           -9.5000000e+00,  1.2734375e+00,  3.6875000e+00],
          [ 5.4687500e+00,  1.1375000e+01,  6.3750000e+00,
            1.2000000e+01,  2.1875000e-01, -4.6875000e-01],
          [-1.2500000e+01,  9.5000000e+00,  8.9375000e+00,
            2.0500000e+01, -1.1125000e+01, -1.6875000e+01],
          [ 3.6750000e+01, -1.4062500e-01,  3.8000000e+01,
            2.4375000e+00,  3.0125000e+01,  7.4375000e+00]],

         [[ 3.1250000e+00,  6.8125000e+00,  2.0125000e+01,
            1.3937500e+01, -1.3125000e+01, -7.9375000e+00],
          [ 7.8125000e+00, -1.5562500e+01, -2.1250000e+01,
            1.3625000e+01, -1.0125000e+01, -3.6562500e+00],
          [-2.3437500e-02,  1.0468750e+00, -1.2937500e+01,
            2.0500000e+01,  1.8500000e+01,  8.3125000e+00],
          [ 1.5500000e+01, -8.8750000e+00,  9.3750000e+00,
            1.8750000e+00,  1.7250000e+01,  8.4375000e-01]],

         [[-1.0000000e+01, -1.0500000e+01, -4.2500000e+00,
           -2.0937500e+00,  2.4125000e+01,  3.7968750e+00],
          [-7.5000000e-01,  1.8375000e+01, -3.8125000e+00,
           -7.8125000e-02, -1.0000000e+00, -7.3125000e+00],
          [ 2.2750000e+01,  1.1875000e+01,  2.1093750e+00,
            2.0750000e+01,  8.5000000e+00,  2.8750000e+01],
          [ 7.4062500e+00, -9.2500000e+00, -5.0625000e+00,
            3.8437500e+00,  5.9375000e+00,  1.0562500e+01]],

         [[-6.1875000e+00, -1.4250000e+01,  2.1875000e+00,
           -2.9375000e+00, -3.1250000e-01, -2.1093750e+00],
          [-3.5312500e+00, -2.1500000e+01, -3.1250000e-01,
           -5.7500000e+00, -1.4750000e+01,  2.4000000e+01],
          [-3.2500000e+01, -9.1875000e+00, -5.4375000e+00,
           -4.7187500e+00, -1.7625000e+01,  1.3937500e+01],
          [-3.1562500e+00, -8.3750000e+00,  5.4687500e+00,
           -2.1625000e+01, -1.3500000e+01,  2.3125000e+00]]],


        [[[ 2.2000000e+01,  3.9000000e+01, -5.6250000e+00,
            2.3750000e+01, -1.1250000e+01,  3.0750000e+01],
          [ 5.8125000e+00, -1.0625000e+01,  6.5312500e+00,
           -2.6125000e+01,  6.7812500e+00,  1.8000000e+01],
          [-1.3187500e+01,  2.8750000e+01, -1.4500000e+01,
           -5.1000000e+01, -2.3750000e+01,  8.6875000e+00],
          [-1.3937500e+01,  6.8125000e+00, -1.5687500e+01,
           -1.2187500e+01,  3.1875000e+00,  8.9375000e+00]],

         [[-2.0000000e+01,  1.8750000e+01, -1.5625000e+01,
           -1.2875000e+01, -4.5312500e+00,  4.4500000e+01],
          [ 1.2875000e+01, -1.5937500e+01, -1.8000000e+01,
            1.7000000e+01, -3.3125000e+00,  1.1625000e+01],
          [-2.7000000e+01,  2.3750000e+00, -1.9125000e+01,
           -2.2750000e+01,  1.1562500e+01,  2.5125000e+01],
          [-1.7000000e+01, -1.0875000e+01,  1.1250000e+01,
            1.2500000e-01,  1.9000000e+01,  1.0250000e+01]],

         [[-1.4750000e+01, -3.8750000e+00,  2.8000000e+01,
            2.5625000e+01,  2.1000000e+01, -8.3125000e+00],
          [-2.6125000e+01,  6.0312500e+00,  4.0000000e+00,
           -1.6000000e+01, -1.1562500e+01, -2.6250000e+01],
          [ 8.1250000e+00, -1.5937500e+01, -1.4062500e+01,
           -3.4375000e+00,  2.2000000e+01, -1.0437500e+01],
          [-2.5625000e+01, -9.3750000e+00, -4.0750000e+01,
           -1.8250000e+01, -2.0750000e+01,  3.1750000e+01]],

         [[-5.0000000e+00,  2.0250000e+01, -2.9000000e+01,
            1.6250000e+01, -1.8000000e+01,  7.8125000e-01],
          [-5.8125000e+00,  4.8750000e+00, -4.6000000e+01,
           -1.6500000e+01,  1.4062500e+01,  2.3625000e+01],
          [-4.6875000e+00, -3.1000000e+01, -1.8500000e+01,
            2.0125000e+01, -2.6250000e+01, -5.6250000e+00],
          [-6.3125000e+00, -2.5125000e+01, -1.4875000e+01,
           -5.1250000e+00,  3.3000000e+01, -9.1250000e+00]],

         [[ 9.3125000e+00, -1.2250000e+01, -2.1000000e+01,
           -1.6000000e+01,  4.0750000e+01,  2.5500000e+01],
          [ 6.9687500e+00,  2.8750000e+01,  3.6250000e+00,
           -1.0625000e+01, -2.4750000e+01, -2.5000000e+01],
          [ 1.5875000e+01, -1.4250000e+01,  2.3000000e+01,
           -1.7875000e+01,  4.1000000e+01,  1.1250000e+01],
          [ 2.2250000e+01,  1.9000000e+01, -4.7750000e+01,
            3.2750000e+01, -1.5312500e+01, -4.1500000e+01]]],


        [[[ 9.2500000e+00,  1.6250000e+01, -3.0000000e+01,
            2.2656250e+00, -6.8125000e+00,  8.1250000e+00],
          [-1.5000000e+01,  2.3437500e+00, -2.2968750e+00,
            6.1875000e+00, -3.6523438e-01,  1.0000000e+01],
          [-1.6500000e+01,  2.9875000e+01,  1.8875000e+01,
            2.5125000e+01,  3.6562500e+00, -1.9765625e+00],
          [-1.5437500e+01, -6.5625000e+00, -4.5937500e+00,
           -1.3687500e+01,  1.1750000e+01,  2.1625000e+01]],

         [[-7.7187500e+00, -9.1875000e+00,  1.6250000e+01,
           -1.1625000e+01, -1.0234375e+00,  1.5875000e+01],
          [ 9.5000000e+00,  5.9687500e+00, -2.5156250e+00,
            1.8000000e+01,  2.3593750e+00, -2.3906250e+00],
          [-1.9125000e+01,  1.2031250e+00,  5.0000000e+00,
            3.1562500e+00, -6.2812500e+00, -5.4062500e+00],
          [ 2.8000000e+01, -6.4843750e-01,  3.7000000e+01,
            1.0437500e+01,  3.0500000e+01,  3.1093750e+00]],

         [[-3.8593750e+00,  8.4375000e+00,  2.8000000e+01,
            2.0875000e+01, -7.2500000e+00, -1.3687500e+01],
          [ 2.0625000e+00, -1.4875000e+01, -1.1937500e+01,
            5.3437500e+00, -1.4062500e+01, -1.3000000e+01],
          [ 2.6406250e+00, -4.6875000e+00, -1.3312500e+01,
            1.8750000e+01,  2.0250000e+01,  2.9531250e+00],
          [ 1.0437500e+01, -8.9375000e+00,  9.0000000e+00,
            6.2500000e-01,  1.7125000e+01,  7.6875000e+00]],

         [[-1.0875000e+01, -4.5000000e+00, -3.0625000e+00,
            2.5781250e+00,  1.2625000e+01,  2.0000000e+00],
          [-6.0937500e+00,  1.9000000e+01, -1.9250000e+01,
           -5.3750000e+00, -1.2968750e+00, -8.7500000e+00],
          [ 1.8125000e+01,  4.6250000e+00, -7.0312500e+00,
            2.6875000e+01, -8.0625000e+00,  2.7000000e+01],
          [-9.3750000e-02, -1.2937500e+01, -6.1250000e+00,
            2.3750000e+00,  9.8125000e+00,  1.1750000e+01]],

         [[-1.0312500e+00, -1.2812500e+01, -6.0312500e+00,
           -2.7343750e+00,  6.5312500e+00,  6.7187500e+00],
          [ 2.9218750e+00, -7.6875000e+00, -6.2812500e+00,
           -9.5000000e+00, -1.8875000e+01,  1.5750000e+01],
          [-2.3000000e+01, -7.0000000e+00,  5.3125000e-01,
           -1.0562500e+01, -3.7968750e+00,  1.5500000e+01],
          [ 5.8437500e+00, -2.2656250e+00, -7.5312500e+00,
           -1.3375000e+01, -1.7375000e+01, -3.8125000e+00]]]]],
      dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.01, atol=0.01', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=0.01, atol=0.01
E           
E           Mismatched elements: 38 / 960 (3.96%)
E           Max absolute difference among violations: 0.15625
E           Max relative difference among violations: 0.5136719
E            ACTUAL: array([[[[[-5.031250e+00, -3.109375e+00, -1.650000e+01, -5.820312e-01,
E                      -1.625000e+00,  1.718750e+00],
E                     [-1.550000e+01,  7.250000e+00, -6.593750e+00,  1.087500e+01,...
E            DESIRED: array([[[[[-5.000000e+00, -3.125000e+00, -1.650000e+01, -5.625000e-01,
E                      -1.609375e+00,  1.687500e+00],
E                     [-1.550000e+01,  7.250000e+00, -6.593750e+00,  1.087500e+01,...

/usr/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________ LaxBackedNumpyTests.testUnion1dSize3 _____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testUnion1dSize3>, shape1 = (2, 1, 4), shape2 = (1, 4)
dtype1 = <class 'numpy.int8'>, dtype2 = <class 'numpy.int32'>, size = 10, fill_value = None

    @jtu.sample_product(
      dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
      dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
      shape1=nonempty_nonscalar_array_shapes,
      shape2=nonempty_nonscalar_array_shapes,
      size=[1, 5, 10],
      fill_value=[None, -1],
    )
    def testUnion1dSize(self, shape1, shape2, dtype1, dtype2, size, fill_value):
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
      def np_fun(arg1, arg2):
        dtype = jnp.promote_types(arg1.dtype, arg2.dtype)
        result = np.union1d(arg1, arg2).astype(dtype)
        fv = result.min() if fill_value is None else fill_value
        if size <= len(result):
          return result[:size]
        else:
          return np.concatenate([result, np.full(size - len(result), fv, result.dtype)])
      def jnp_fun(arg1, arg2):
        return jnp.union1d(arg1, arg2, size=size, fill_value=fill_value)
      with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
>       self._CompileAndCheck(jnp_fun, args_maker)

lax_stablehlo_reference_test.py:748: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1322: in _CompileAndCheck
    self.assertAllClose(python_ans, monitored_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f027355e980>, array([-5, -4, -2, -1,  0,  1,  2,  4, -5, -5], dtype=int32), array([-5, -4, -2, -1,  0,  1,  2,  4,  4, -5], dtype=int32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-07, atol=0', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-07, atol=0
E           
E           Mismatched elements: 1 / 10 (10%)
E           Max absolute difference among violations: 9
E           Max relative difference among violations: 2.25
E            ACTUAL: array([-5, -4, -2, -1,  0,  1,  2,  4, -5, -5], dtype=int32)
E            DESIRED: array([-5, -4, -2, -1,  0,  1,  2,  4,  4, -5], dtype=int32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________ LaxBackedNumpyTests.testUnion1dSize4 _____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testUnion1dSize4>, shape1 = (4,), shape2 = (3, 1)
dtype1 = <class 'numpy.int8'>, dtype2 = <class 'numpy.float32'>, size = 10, fill_value = None

    @jtu.sample_product(
      dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
      dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
      shape1=nonempty_nonscalar_array_shapes,
      shape2=nonempty_nonscalar_array_shapes,
      size=[1, 5, 10],
      fill_value=[None, -1],
    )
    def testUnion1dSize(self, shape1, shape2, dtype1, dtype2, size, fill_value):
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
      def np_fun(arg1, arg2):
        dtype = jnp.promote_types(arg1.dtype, arg2.dtype)
        result = np.union1d(arg1, arg2).astype(dtype)
        fv = result.min() if fill_value is None else fill_value
        if size <= len(result):
          return result[:size]
        else:
          return np.concatenate([result, np.full(size - len(result), fv, result.dtype)])
      def jnp_fun(arg1, arg2):
        return jnp.union1d(arg1, arg2, size=size, fill_value=fill_value)
      with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
>       self._CompileAndCheck(jnp_fun, args_maker)

lax_stablehlo_reference_test.py:748: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1322: in _CompileAndCheck
    self.assertAllClose(python_ans, monitored_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f027355f9c0>, array([-5.279511 , -3.       ,  0.       ,  1.5341507,  4.624709 ,
       -5.279511 , -5.279511 , -5.279511 , -5.279511 , -5.279511 ],
      dtype=float32), array([-5.279511 , -3.       ,  0.       ,  1.5341507,  4.624709 ,
        4.624709 , -5.279511 , -5.279511 , -5.279511 , -5.279511 ],
      dtype=float32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-06, atol=1e-06', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-06, atol=1e-06
E           
E           Mismatched elements: 1 / 10 (10%)
E           Max absolute difference among violations: 9.904221
E           Max relative difference among violations: 2.1415877
E            ACTUAL: array([-5.279511, -3.      ,  0.      ,  1.534151,  4.624709, -5.279511,
E                  -5.279511, -5.279511, -5.279511, -5.279511], dtype=float32)
E            DESIRED: array([-5.279511, -3.      ,  0.      ,  1.534151,  4.624709,  4.624709,
E                  -5.279511, -5.279511, -5.279511, -5.279511], dtype=float32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________ LaxBackedNumpyTests.testUnion1dSize9 _____________________________________________

self = <lax_stablehlo_reference_test.LaxBackedNumpyTests testMethod=testUnion1dSize9>, shape1 = (4,), shape2 = (4,)
dtype1 = <class 'numpy.int8'>, dtype2 = <class 'numpy.int32'>, size = 10, fill_value = None

    @jtu.sample_product(
      dtype1=[s for s in default_dtypes if s != jnp.bfloat16],
      dtype2=[s for s in default_dtypes if s != jnp.bfloat16],
      shape1=nonempty_nonscalar_array_shapes,
      shape2=nonempty_nonscalar_array_shapes,
      size=[1, 5, 10],
      fill_value=[None, -1],
    )
    def testUnion1dSize(self, shape1, shape2, dtype1, dtype2, size, fill_value):
      rng = jtu.rand_default(self.rng())
      args_maker = lambda: [rng(shape1, dtype1), rng(shape2, dtype2)]
      def np_fun(arg1, arg2):
        dtype = jnp.promote_types(arg1.dtype, arg2.dtype)
        result = np.union1d(arg1, arg2).astype(dtype)
        fv = result.min() if fill_value is None else fill_value
        if size <= len(result):
          return result[:size]
        else:
          return np.concatenate([result, np.full(size - len(result), fv, result.dtype)])
      def jnp_fun(arg1, arg2):
        return jnp.union1d(arg1, arg2, size=size, fill_value=fill_value)
      with jtu.strict_promotion_if_dtypes_match([dtype1, dtype2]):
        self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker)
>       self._CompileAndCheck(jnp_fun, args_maker)

lax_stablehlo_reference_test.py:748: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
env/lib/python3.11/site-packages/jax/_src/test_util.py:1322: in _CompileAndCheck
    self.assertAllClose(python_ans, monitored_ans, check_dtypes=check_dtypes,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1265: in assertAllClose
    self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
env/lib/python3.11/site-packages/jax/_src/test_util.py:1230: in assertArraysAllClose
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)
env/lib/python3.11/site-packages/jax/_src/public_test_util.py:128: in _assert_numpy_allclose
    np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x7f0272b734c0>, array([-3, -2,  0,  1,  2,  5, -3, -3, -3, -3], dtype=int32), array([-3, -2,  0,  1,  2,  5,  5, -3, -3, -3], dtype=int32))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=1e-07, atol=0', 'strict': False, ...}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Not equal to tolerance rtol=1e-07, atol=0
E           
E           Mismatched elements: 1 / 10 (10%)
E           Max absolute difference among violations: 8
E           Max relative difference among violations: 1.6
E            ACTUAL: array([-3, -2,  0,  1,  2,  5, -3, -3, -3, -3], dtype=int32)
E            DESIRED: array([-3, -2,  0,  1,  2,  5,  5, -3, -3, -3], dtype=int32)

/usr/lib/python3.11/contextlib.py:81: AssertionError
=================================================== short test summary info ===================================================
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testConvolutions2 - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 2 / 7 (28.6%)
Max absolute difference among violations: 0.09375
Max relative difference among violations: 0.03225806
 ACTUAL: array([  3.    ,  11.375 ,  21.5   ,  -4.6875,  43.25  , -19.875 ,
        -8.9375], dtype=float32)
 DESIRED: array([  2.90625,  11.375  ,  21.5    ,  -4.625  ,  43.25   , -19.875  ,
        -9.     ], dtype=float32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testConvolutions4 - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 3 / 12 (25%)
Max absolute difference among violations: 0.1875
Max relative difference among violations: 0.03515625
 ACTUAL: array([-30.875   , -43.      , -56.25    , -40.5     , -34.5     ,
        -9.5     ,   0.964844,   4.625   ,  14.3125  , -36.75    ,
        20.375   , -65.      ], dtype=float32)
 DESIRED: array([-30.75  , -43.    , -56.5   , -40.5   , -34.5   ,  -9.3125,
         1.    ,   4.5   ,  14.1875, -36.75  ,  20.5   , -65.    ],
      dtype=float32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testConvolutions9 - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 1 / 23 (4.35%)
Max absolute difference among violations: 0.25
Max relative difference among violations: 0.01111111
 ACTUAL: array([  0.056885,  15.3125  , -22.125   ,  19.125   , -22.125   ,
        49.25    , -42.      ,  54.75    , -36.      ,  15.9375  ,
       -22.625   ,  20.625   , -33.      ,  22.25    , -37.      ,...
 DESIRED: array([  0.056885,  15.3125  , -22.25    ,  19.125   , -22.125   ,
        49.25    , -42.      ,  54.75    , -36.      ,  15.875   ,
       -22.5     ,  20.625   , -33.25    ,  22.5     , -36.75    ,...
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testConvolutionsPreferredElementType4 - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 1 / 12 (8.33%)
Max absolute difference among violations: 0.125
Max relative difference among violations: 0.08510638
 ACTUAL: array([  0.882812, -29.125   , -12.875   ,  28.625   ,  18.625   ,
       -14.5625  ,   1.59375 , -25.75    ,  -1.429688,  20.125   ,
        10.5     , -13.125   ], dtype=float32)
 DESIRED: array([  0.875   , -29.125   , -12.9375  ,  28.75    ,  18.5     ,
       -14.5625  ,   1.46875 , -25.875   ,  -1.453125,  20.      ,
        10.4375  , -13.125   ], dtype=float32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testConvolutionsPreferredElementType9 - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 2 / 23 (8.7%)
Max absolute difference among violations: 0.1875
Max relative difference among violations: 0.03550296
 ACTUAL: array([ -0.316406,   1.101562,  -4.1875  ,  -3.265625,   6.375   ,
        16.375   ,   9.      , -26.75    , -10.8125  ,  24.5     ,
        10.4375  , -15.625   , -11.125   ,   5.09375 ,  -0.53125 ,...
 DESIRED: array([ -0.316406,   1.09375 ,  -4.21875 ,  -3.25    ,   6.375   ,
        16.375   ,   9.      , -26.875   , -10.8125  ,  24.5     ,
        10.4375  , -15.625   , -11.125   ,   5.28125 ,  -0.546875,...
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testFlatNonzeroSize3 - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 1 / 1 (100%)
Max absolute difference among violations: 3
Max relative difference among violations: 3.
 ACTUAL: array([2], dtype=int32)
 DESIRED: array([-1], dtype=int32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testIntegerPowerOverflow1 - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 1 / 1 (100%)
Max absolute difference among violations: 2
Max relative difference among violations: 2.
 ACTUAL: array(1, dtype=int32)
 DESIRED: array(-1, dtype=int32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testIsClose - jaxlib.xla_extension.XlaRuntimeError: UNIMPLEMENTED: Unsupported op: %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{replicated}"} : (tensor<2xui32>) -> tensor<2xui32>
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testNonScalarRepeats1 - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 8 / 12 (66.7%)
Max absolute difference among violations: 3
Max relative difference among violations: 1.5
 ACTUAL: array([1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=int32)
 DESIRED: array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=int32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testNonzeroSize3 - ValueError: Calling nonzero on 0d arrays is not allowed. Use jnp.atleast_1d(scalar).nonzero() instead.
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testNonzeroSize5 - ValueError: Calling nonzero on 0d arrays is not allowed. Use jnp.atleast_1d(scalar).nonzero() instead.
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testTensordot5 - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 11 / 60 (18.3%)
Max absolute difference among violations: 0.875
Max relative difference among violations: 0.03888889
 ACTUAL: array([[[[-22.125   ,  64.5     ,  58.25    ,  28.875   ,  46.5     ,
           73.5     ],
         [ 46.5     , -13.0625  , -79.5     ,  26.75    ,  51.      ,...
 DESIRED: array([[[[-22.25   ,  64.5    ,  58.25   ,  28.625  ,  47.     ,
           73.5    ],
         [ 46.75   , -13.     , -80.     ,  26.75   ,  50.75   ,...
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testTensordot8 - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 38 / 960 (3.96%)
Max absolute difference among violations: 0.15625
Max relative difference among violations: 0.5136719
 ACTUAL: array([[[[[-5.031250e+00, -3.109375e+00, -1.650000e+01, -5.820312e-01,
           -1.625000e+00,  1.718750e+00],
          [-1.550000e+01,  7.250000e+00, -6.593750e+00,  1.087500e+01,...
 DESIRED: array([[[[[-5.000000e+00, -3.125000e+00, -1.650000e+01, -5.625000e-01,
           -1.609375e+00,  1.687500e+00],
          [-1.550000e+01,  7.250000e+00, -6.593750e+00,  1.087500e+01,...
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testUnion1dSize3 - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 1 / 10 (10%)
Max absolute difference among violations: 9
Max relative difference among violations: 2.25
 ACTUAL: array([-5, -4, -2, -1,  0,  1,  2,  4, -5, -5], dtype=int32)
 DESIRED: array([-5, -4, -2, -1,  0,  1,  2,  4,  4, -5], dtype=int32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testUnion1dSize4 - AssertionError: 
Not equal to tolerance rtol=1e-06, atol=1e-06

Mismatched elements: 1 / 10 (10%)
Max absolute difference among violations: 9.904221
Max relative difference among violations: 2.1415877
 ACTUAL: array([-5.279511, -3.      ,  0.      ,  1.534151,  4.624709, -5.279511,
       -5.279511, -5.279511, -5.279511, -5.279511], dtype=float32)
 DESIRED: array([-5.279511, -3.      ,  0.      ,  1.534151,  4.624709,  4.624709,
       -5.279511, -5.279511, -5.279511, -5.279511], dtype=float32)
FAILED lax_stablehlo_reference_test.py::LaxBackedNumpyTests::testUnion1dSize9 - AssertionError: 
Not equal to tolerance rtol=1e-07, atol=0

Mismatched elements: 1 / 10 (10%)
Max absolute difference among violations: 8
Max relative difference among violations: 1.6
 ACTUAL: array([-3, -2,  0,  1,  2,  5, -3, -3, -3, -3], dtype=int32)
 DESIRED: array([-3, -2,  0,  1,  2,  5,  5, -3, -3, -3], dtype=int32)
================================== 16 failed, 1676 passed, 58 skipped in 1035.19s (0:17:15) ===================================
