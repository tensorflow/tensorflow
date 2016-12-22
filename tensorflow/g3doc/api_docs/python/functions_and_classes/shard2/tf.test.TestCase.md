Base class for tests that need to test TensorFlow.
- - -

#### `tf.test.TestCase.__call__(*args, **kwds)` {#TestCase.__call__}




- - -

#### `tf.test.TestCase.__eq__(other)` {#TestCase.__eq__}




- - -

#### `tf.test.TestCase.__hash__()` {#TestCase.__hash__}




- - -

#### `tf.test.TestCase.__init__(methodName='runTest')` {#TestCase.__init__}




- - -

#### `tf.test.TestCase.__ne__(other)` {#TestCase.__ne__}




- - -

#### `tf.test.TestCase.__repr__()` {#TestCase.__repr__}




- - -

#### `tf.test.TestCase.__str__()` {#TestCase.__str__}




- - -

#### `tf.test.TestCase.addCleanup(function, *args, **kwargs)` {#TestCase.addCleanup}

Add a function, with arguments, to be called when the test is
completed. Functions added are called on a LIFO basis and are
called after tearDown on test failure or success.

Cleanup items are called even if setUp fails (unlike tearDown).


- - -

#### `tf.test.TestCase.addTypeEqualityFunc(typeobj, function)` {#TestCase.addTypeEqualityFunc}

Add a type specific assertEqual style function to compare a type.

This method is for use by TestCase subclasses that need to register
their own type equality functions to provide nicer error messages.

##### Args:


*  <b>`typeobj`</b>: The data type to call this function on when both values
            are of the same type in assertEqual().
*  <b>`function`</b>: The callable taking two arguments and an optional
            msg= argument that raises self.failureException with a
            useful error message when the two arguments are not equal.


- - -

#### `tf.test.TestCase.assertAllClose(a, b, rtol=1e-06, atol=1e-06)` {#TestCase.assertAllClose}

Asserts that two numpy arrays have near values.

##### Args:


*  <b>`a`</b>: a numpy ndarray or anything can be converted to one.
*  <b>`b`</b>: a numpy ndarray or anything can be converted to one.
*  <b>`rtol`</b>: relative tolerance
*  <b>`atol`</b>: absolute tolerance


- - -

#### `tf.test.TestCase.assertAllCloseAccordingToType(a, b, rtol=1e-06, atol=1e-06)` {#TestCase.assertAllCloseAccordingToType}

Like assertAllClose, but also suitable for comparing fp16 arrays.

In particular, the tolerance is reduced to 1e-3 if at least
one of the arguments is of type float16.

##### Args:


*  <b>`a`</b>: a numpy ndarray or anything can be converted to one.
*  <b>`b`</b>: a numpy ndarray or anything can be converted to one.
*  <b>`rtol`</b>: relative tolerance
*  <b>`atol`</b>: absolute tolerance


- - -

#### `tf.test.TestCase.assertAllEqual(a, b)` {#TestCase.assertAllEqual}

Asserts that two numpy arrays have the same values.

##### Args:


*  <b>`a`</b>: a numpy ndarray or anything can be converted to one.
*  <b>`b`</b>: a numpy ndarray or anything can be converted to one.


- - -

#### `tf.test.TestCase.assertAlmostEqual(first, second, places=None, msg=None, delta=None)` {#TestCase.assertAlmostEqual}

Fail if the two objects are unequal as determined by their
difference rounded to the given number of decimal places
(default 7) and comparing to zero, or by comparing that the
between the two objects is more than the given delta.

Note that decimal places (from zero) are usually not the same
as significant digits (measured from the most signficant digit).

If the two objects compare equal then they will automatically
compare almost equal.


- - -

#### `tf.test.TestCase.assertAlmostEquals(first, second, places=None, msg=None, delta=None)` {#TestCase.assertAlmostEquals}

Fail if the two objects are unequal as determined by their
difference rounded to the given number of decimal places
(default 7) and comparing to zero, or by comparing that the
between the two objects is more than the given delta.

Note that decimal places (from zero) are usually not the same
as significant digits (measured from the most signficant digit).

If the two objects compare equal then they will automatically
compare almost equal.


- - -

#### `tf.test.TestCase.assertArrayNear(farray1, farray2, err)` {#TestCase.assertArrayNear}

Asserts that two float arrays are near each other.

Checks that for all elements of farray1 and farray2
|f1 - f2| < err.  Asserts a test failure if not.

##### Args:


*  <b>`farray1`</b>: a list of float values.
*  <b>`farray2`</b>: a list of float values.
*  <b>`err`</b>: a float value.


- - -

#### `tf.test.TestCase.assertDeviceEqual(device1, device2)` {#TestCase.assertDeviceEqual}

Asserts that the two given devices are the same.

##### Args:


*  <b>`device1`</b>: A string device name or TensorFlow `DeviceSpec` object.
*  <b>`device2`</b>: A string device name or TensorFlow `DeviceSpec` object.


- - -

#### `tf.test.TestCase.assertDictContainsSubset(expected, actual, msg=None)` {#TestCase.assertDictContainsSubset}

Checks whether actual is a superset of expected.


- - -

#### `tf.test.TestCase.assertDictEqual(d1, d2, msg=None)` {#TestCase.assertDictEqual}




- - -

#### `tf.test.TestCase.assertEqual(first, second, msg=None)` {#TestCase.assertEqual}

Fail if the two objects are unequal as determined by the '=='
operator.


- - -

#### `tf.test.TestCase.assertEquals(first, second, msg=None)` {#TestCase.assertEquals}

Fail if the two objects are unequal as determined by the '=='
operator.


- - -

#### `tf.test.TestCase.assertFalse(expr, msg=None)` {#TestCase.assertFalse}

Check that the expression is false.


- - -

#### `tf.test.TestCase.assertGreater(a, b, msg=None)` {#TestCase.assertGreater}

Just like self.assertTrue(a > b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertGreaterEqual(a, b, msg=None)` {#TestCase.assertGreaterEqual}

Just like self.assertTrue(a >= b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertIn(member, container, msg=None)` {#TestCase.assertIn}

Just like self.assertTrue(a in b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertIs(expr1, expr2, msg=None)` {#TestCase.assertIs}

Just like self.assertTrue(a is b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertIsInstance(obj, cls, msg=None)` {#TestCase.assertIsInstance}

Same as self.assertTrue(isinstance(obj, cls)), with a nicer
default message.


- - -

#### `tf.test.TestCase.assertIsNone(obj, msg=None)` {#TestCase.assertIsNone}

Same as self.assertTrue(obj is None), with a nicer default message.


- - -

#### `tf.test.TestCase.assertIsNot(expr1, expr2, msg=None)` {#TestCase.assertIsNot}

Just like self.assertTrue(a is not b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertIsNotNone(obj, msg=None)` {#TestCase.assertIsNotNone}

Included for symmetry with assertIsNone.


- - -

#### `tf.test.TestCase.assertItemsEqual(expected_seq, actual_seq, msg=None)` {#TestCase.assertItemsEqual}

An unordered sequence specific comparison. It asserts that
actual_seq and expected_seq have the same element counts.
Equivalent to::

    self.assertEqual(Counter(iter(actual_seq)),
                     Counter(iter(expected_seq)))

Asserts that each element has the same count in both sequences.

##### Example:

    - [0, 1, 1] and [1, 0, 1] compare equal.
    - [0, 0, 1] and [0, 1] compare unequal.


- - -

#### `tf.test.TestCase.assertLess(a, b, msg=None)` {#TestCase.assertLess}

Just like self.assertTrue(a < b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertLessEqual(a, b, msg=None)` {#TestCase.assertLessEqual}

Just like self.assertTrue(a <= b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertListEqual(list1, list2, msg=None)` {#TestCase.assertListEqual}

A list-specific equality assertion.

##### Args:


*  <b>`list1`</b>: The first list to compare.
*  <b>`list2`</b>: The second list to compare.
*  <b>`msg`</b>: Optional message to use on failure instead of a list of
            differences.


- - -

#### `tf.test.TestCase.assertMultiLineEqual(first, second, msg=None)` {#TestCase.assertMultiLineEqual}

Assert that two multi-line strings are equal.


- - -

#### `tf.test.TestCase.assertNDArrayNear(ndarray1, ndarray2, err)` {#TestCase.assertNDArrayNear}

Asserts that two numpy arrays have near values.

##### Args:


*  <b>`ndarray1`</b>: a numpy ndarray.
*  <b>`ndarray2`</b>: a numpy ndarray.
*  <b>`err`</b>: a float. The maximum absolute difference allowed.


- - -

#### `tf.test.TestCase.assertNear(f1, f2, err, msg=None)` {#TestCase.assertNear}

Asserts that two floats are near each other.

Checks that |f1 - f2| < err and asserts a test failure
if not.

##### Args:


*  <b>`f1`</b>: A float value.
*  <b>`f2`</b>: A float value.
*  <b>`err`</b>: A float value.
*  <b>`msg`</b>: An optional string message to append to the failure message.


- - -

#### `tf.test.TestCase.assertNotAlmostEqual(first, second, places=None, msg=None, delta=None)` {#TestCase.assertNotAlmostEqual}

Fail if the two objects are equal as determined by their
difference rounded to the given number of decimal places
(default 7) and comparing to zero, or by comparing that the
between the two objects is less than the given delta.

Note that decimal places (from zero) are usually not the same
as significant digits (measured from the most signficant digit).

Objects that are equal automatically fail.


- - -

#### `tf.test.TestCase.assertNotAlmostEquals(first, second, places=None, msg=None, delta=None)` {#TestCase.assertNotAlmostEquals}

Fail if the two objects are equal as determined by their
difference rounded to the given number of decimal places
(default 7) and comparing to zero, or by comparing that the
between the two objects is less than the given delta.

Note that decimal places (from zero) are usually not the same
as significant digits (measured from the most signficant digit).

Objects that are equal automatically fail.


- - -

#### `tf.test.TestCase.assertNotEqual(first, second, msg=None)` {#TestCase.assertNotEqual}

Fail if the two objects are equal as determined by the '!='
operator.


- - -

#### `tf.test.TestCase.assertNotEquals(first, second, msg=None)` {#TestCase.assertNotEquals}

Fail if the two objects are equal as determined by the '!='
operator.


- - -

#### `tf.test.TestCase.assertNotIn(member, container, msg=None)` {#TestCase.assertNotIn}

Just like self.assertTrue(a not in b), but with a nicer default message.


- - -

#### `tf.test.TestCase.assertNotIsInstance(obj, cls, msg=None)` {#TestCase.assertNotIsInstance}

Included for symmetry with assertIsInstance.


- - -

#### `tf.test.TestCase.assertNotRegexpMatches(text, unexpected_regexp, msg=None)` {#TestCase.assertNotRegexpMatches}

Fail the test if the text matches the regular expression.


- - -

#### `tf.test.TestCase.assertProtoEquals(expected_message_maybe_ascii, message)` {#TestCase.assertProtoEquals}

Asserts that message is same as parsed expected_message_ascii.

Creates another prototype of message, reads the ascii message into it and
then compares them using self._AssertProtoEqual().

##### Args:


*  <b>`expected_message_maybe_ascii`</b>: proto message in original or ascii form
*  <b>`message`</b>: the message to validate


- - -

#### `tf.test.TestCase.assertProtoEqualsVersion(expected, actual, producer=20, min_consumer=0)` {#TestCase.assertProtoEqualsVersion}




- - -

#### `tf.test.TestCase.assertRaises(excClass, callableObj=None, *args, **kwargs)` {#TestCase.assertRaises}

Fail unless an exception of class excClass is raised
by callableObj when invoked with arguments args and keyword
arguments kwargs. If a different type of exception is
raised, it will not be caught, and the test case will be
deemed to have suffered an error, exactly as for an
unexpected exception.

If called with callableObj omitted or None, will return a
context object used like this::

     with self.assertRaises(SomeException):
         do_something()

The context manager keeps a reference to the exception as
the 'exception' attribute. This allows you to inspect the
exception after the assertion::

    with self.assertRaises(SomeException) as cm:
        do_something()
    the_exception = cm.exception
    self.assertEqual(the_exception.error_code, 3)


- - -

#### `tf.test.TestCase.assertRaisesOpError(expected_err_re_or_predicate)` {#TestCase.assertRaisesOpError}




- - -

#### `tf.test.TestCase.assertRaisesRegexp(expected_exception, expected_regexp, callable_obj=None, *args, **kwargs)` {#TestCase.assertRaisesRegexp}

Asserts that the message in a raised exception matches a regexp.

##### Args:


*  <b>`expected_exception`</b>: Exception class expected to be raised.
*  <b>`expected_regexp`</b>: Regexp (re pattern object or string) expected
            to be found in error message.
*  <b>`callable_obj`</b>: Function to be called.
*  <b>`args`</b>: Extra args.
*  <b>`kwargs`</b>: Extra kwargs.


- - -

#### `tf.test.TestCase.assertRaisesWithPredicateMatch(exception_type, expected_err_re_or_predicate)` {#TestCase.assertRaisesWithPredicateMatch}

Returns a context manager to enclose code expected to raise an exception.

If the exception is an OpError, the op stack is also included in the message
predicate search.

##### Args:


*  <b>`exception_type`</b>: The expected type of exception that should be raised.
*  <b>`expected_err_re_or_predicate`</b>: If this is callable, it should be a function
    of one argument that inspects the passed-in exception and
    returns True (success) or False (please fail the test). Otherwise, the
    error message is expected to match this regular expression partially.

##### Returns:

  A context manager to surround code that is expected to raise an
  exception.


- - -

#### `tf.test.TestCase.assertRegexpMatches(text, expected_regexp, msg=None)` {#TestCase.assertRegexpMatches}

Fail the test unless the text matches the regular expression.


- - -

#### `tf.test.TestCase.assertSequenceEqual(seq1, seq2, msg=None, seq_type=None)` {#TestCase.assertSequenceEqual}

An equality assertion for ordered sequences (like lists and tuples).

For the purposes of this function, a valid ordered sequence type is one
which can be indexed, has a length, and has an equality operator.

##### Args:


*  <b>`seq1`</b>: The first sequence to compare.
*  <b>`seq2`</b>: The second sequence to compare.
*  <b>`seq_type`</b>: The expected datatype of the sequences, or None if no
            datatype should be enforced.
*  <b>`msg`</b>: Optional message to use on failure instead of a list of
            differences.


- - -

#### `tf.test.TestCase.assertSetEqual(set1, set2, msg=None)` {#TestCase.assertSetEqual}

A set-specific equality assertion.

##### Args:


*  <b>`set1`</b>: The first set to compare.
*  <b>`set2`</b>: The second set to compare.
*  <b>`msg`</b>: Optional message to use on failure instead of a list of
            differences.

assertSetEqual uses ducktyping to support different types of sets, and
is optimized for sets specifically (parameters must support a
difference method).


- - -

#### `tf.test.TestCase.assertShapeEqual(np_array, tf_tensor)` {#TestCase.assertShapeEqual}

Asserts that a Numpy ndarray and a TensorFlow tensor have the same shape.

##### Args:


*  <b>`np_array`</b>: A Numpy ndarray or Numpy scalar.
*  <b>`tf_tensor`</b>: A Tensor.

##### Raises:


*  <b>`TypeError`</b>: If the arguments have the wrong type.


- - -

#### `tf.test.TestCase.assertStartsWith(actual, expected_start, msg=None)` {#TestCase.assertStartsWith}

Assert that actual.startswith(expected_start) is True.

##### Args:


*  <b>`actual`</b>: str
*  <b>`expected_start`</b>: str
*  <b>`msg`</b>: Optional message to report on failure.


- - -

#### `tf.test.TestCase.assertTrue(expr, msg=None)` {#TestCase.assertTrue}

Check that the expression is true.


- - -

#### `tf.test.TestCase.assertTupleEqual(tuple1, tuple2, msg=None)` {#TestCase.assertTupleEqual}

A tuple-specific equality assertion.

##### Args:


*  <b>`tuple1`</b>: The first tuple to compare.
*  <b>`tuple2`</b>: The second tuple to compare.
*  <b>`msg`</b>: Optional message to use on failure instead of a list of
            differences.


- - -

#### `tf.test.TestCase.assert_(expr, msg=None)` {#TestCase.assert_}

Check that the expression is true.


- - -

#### `tf.test.TestCase.checkedThread(target, args=None, kwargs=None)` {#TestCase.checkedThread}

Returns a Thread wrapper that asserts 'target' completes successfully.

This method should be used to create all threads in test cases, as
otherwise there is a risk that a thread will silently fail, and/or
assertions made in the thread will not be respected.

##### Args:


*  <b>`target`</b>: A callable object to be executed in the thread.
*  <b>`args`</b>: The argument tuple for the target invocation. Defaults to ().
*  <b>`kwargs`</b>: A dictionary of keyword arguments for the target invocation.
    Defaults to {}.

##### Returns:

  A wrapper for threading.Thread that supports start() and join() methods.


- - -

#### `tf.test.TestCase.countTestCases()` {#TestCase.countTestCases}




- - -

#### `tf.test.TestCase.debug()` {#TestCase.debug}

Run the test without collecting errors in a TestResult


- - -

#### `tf.test.TestCase.defaultTestResult()` {#TestCase.defaultTestResult}




- - -

#### `tf.test.TestCase.doCleanups()` {#TestCase.doCleanups}

Execute all cleanup functions. Normally called for you after
tearDown.


- - -

#### `tf.test.TestCase.fail(msg=None)` {#TestCase.fail}

Fail immediately, with the given message.


- - -

#### `tf.test.TestCase.failIf(*args, **kwargs)` {#TestCase.failIf}




- - -

#### `tf.test.TestCase.failIfAlmostEqual(*args, **kwargs)` {#TestCase.failIfAlmostEqual}




- - -

#### `tf.test.TestCase.failIfEqual(*args, **kwargs)` {#TestCase.failIfEqual}




- - -

#### `tf.test.TestCase.failUnless(*args, **kwargs)` {#TestCase.failUnless}




- - -

#### `tf.test.TestCase.failUnlessAlmostEqual(*args, **kwargs)` {#TestCase.failUnlessAlmostEqual}




- - -

#### `tf.test.TestCase.failUnlessEqual(*args, **kwargs)` {#TestCase.failUnlessEqual}




- - -

#### `tf.test.TestCase.failUnlessRaises(*args, **kwargs)` {#TestCase.failUnlessRaises}




- - -

#### `tf.test.TestCase.get_temp_dir()` {#TestCase.get_temp_dir}




- - -

#### `tf.test.TestCase.id()` {#TestCase.id}




- - -

#### `tf.test.TestCase.run(result=None)` {#TestCase.run}




- - -

#### `tf.test.TestCase.setUp()` {#TestCase.setUp}




- - -

#### `tf.test.TestCase.setUpClass(cls)` {#TestCase.setUpClass}

Hook method for setting up class fixture before running tests in the class.


- - -

#### `tf.test.TestCase.shortDescription()` {#TestCase.shortDescription}

Returns a one-line description of the test, or None if no
description has been provided.

The default implementation of this method returns the first line of
the specified test method's docstring.


- - -

#### `tf.test.TestCase.skipTest(reason)` {#TestCase.skipTest}

Skip this test.


- - -

#### `tf.test.TestCase.tearDown()` {#TestCase.tearDown}




- - -

#### `tf.test.TestCase.tearDownClass(cls)` {#TestCase.tearDownClass}

Hook method for deconstructing the class fixture after running all tests in the class.


- - -

#### `tf.test.TestCase.test_session(graph=None, config=None, use_gpu=False, force_gpu=False)` {#TestCase.test_session}

Returns a TensorFlow Session for use in executing tests.

This method should be used for all functional tests.

Use the `use_gpu` and `force_gpu` options to control where ops are run. If
`force_gpu` is True, all ops are pinned to `/gpu:0`. Otherwise, if `use_gpu`
is True, TensorFlow tries to run as many ops on the GPU as possible. If both
`force_gpu and `use_gpu` are False, all ops are pinned to the CPU.

Example:

  class MyOperatorTest(test_util.TensorFlowTestCase):
    def testMyOperator(self):
      with self.test_session(use_gpu=True):
        valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = MyOperator(valid_input).eval()
        self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
        invalid_input = [-1.0, 2.0, 7.0]
        with self.assertRaisesOpError("negative input not supported"):
          MyOperator(invalid_input).eval()

##### Args:


*  <b>`graph`</b>: Optional graph to use during the returned session.
*  <b>`config`</b>: An optional config_pb2.ConfigProto to use to configure the
    session.
*  <b>`use_gpu`</b>: If True, attempt to run as many ops as possible on GPU.
*  <b>`force_gpu`</b>: If True, pin all ops to `/gpu:0`.

##### Returns:

  A Session object that should be used as a context manager to surround
  the graph building and execution code in a test case.


