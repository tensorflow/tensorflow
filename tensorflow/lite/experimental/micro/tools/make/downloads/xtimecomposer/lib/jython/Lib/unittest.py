#!/usr/bin/env python
'''
Python unit testing framework, based on Erich Gamma's JUnit and Kent Beck's
Smalltalk testing framework.

This module contains the core framework classes that form the basis of
specific test cases and suites (TestCase, TestSuite etc.), and also a
text-based utility class for running the tests and reporting the results
 (TextTestRunner).

Simple usage:

    import unittest

    class IntegerArithmenticTestCase(unittest.TestCase):
        def testAdd(self):  ## test method names begin 'test*'
            self.assertEquals((1 + 2), 3)
            self.assertEquals(0 + 1, 1)
        def testMultiply(self):
            self.assertEquals((0 * 10), 0)
            self.assertEquals((5 * 8), 40)

    if __name__ == '__main__':
        unittest.main()

Further information is available in the bundled documentation, and from

  http://pyunit.sourceforge.net/

Copyright (c) 1999-2003 Steve Purcell
This module is free software, and you may redistribute it and/or modify
it under the same terms as Python itself, so long as this copyright message
and disclaimer are retained in their original form.

IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF
THIS CODE, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE.  THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THERE IS NO OBLIGATION WHATSOEVER TO PROVIDE MAINTENANCE,
SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
'''

__author__ = "Steve Purcell"
__email__ = "stephen_purcell at yahoo dot com"
__version__ = "#Revision: 1.63 $"[11:-2]

import time
import sys
import traceback
import os
import types

##############################################################################
# Exported classes and functions
##############################################################################
__all__ = ['TestResult', 'TestCase', 'TestSuite', 'TextTestRunner',
           'TestLoader', 'FunctionTestCase', 'main', 'defaultTestLoader']

# Expose obsolete functions for backwards compatibility
__all__.extend(['getTestCaseNames', 'makeSuite', 'findTestCases'])


##############################################################################
# Backward compatibility
##############################################################################
if sys.version_info[:2] < (2, 2):
    False, True = 0, 1
    def isinstance(obj, clsinfo):
        import __builtin__
        if type(clsinfo) in (tuple, list):
            for cls in clsinfo:
                if cls is type: cls = types.ClassType
                if __builtin__.isinstance(obj, cls):
                    return 1
            return 0
        else: return __builtin__.isinstance(obj, clsinfo)


##############################################################################
# Test framework core
##############################################################################

# All classes defined herein are 'new-style' classes, allowing use of 'super()'
__metaclass__ = type

def _strclass(cls):
    return "%s.%s" % (cls.__module__, cls.__name__)

__unittest = 1

class TestResult:
    """Holder for test result information.

    Test results are automatically managed by the TestCase and TestSuite
    classes, and do not need to be explicitly manipulated by writers of tests.

    Each instance holds the total number of tests run, and collections of
    failures and errors that occurred among those test runs. The collections
    contain tuples of (testcase, exceptioninfo), where exceptioninfo is the
    formatted traceback of the error that occurred.
    """
    def __init__(self):
        self.failures = []
        self.errors = []
        self.testsRun = 0
        self.shouldStop = 0

    def startTest(self, test):
        "Called when the given test is about to be run"
        self.testsRun = self.testsRun + 1

    def stopTest(self, test):
        "Called when the given test has been run"
        pass

    def addError(self, test, err):
        """Called when an error has occurred. 'err' is a tuple of values as
        returned by sys.exc_info().
        """
        self.errors.append((test, self._exc_info_to_string(err, test)))

    def addFailure(self, test, err):
        """Called when an error has occurred. 'err' is a tuple of values as
        returned by sys.exc_info()."""
        self.failures.append((test, self._exc_info_to_string(err, test)))

    def addSuccess(self, test):
        "Called when a test has completed successfully"
        pass

    def wasSuccessful(self):
        "Tells whether or not this result was a success"
        return len(self.failures) == len(self.errors) == 0

    def stop(self):
        "Indicates that the tests should be aborted"
        self.shouldStop = True

    def _exc_info_to_string(self, err, test):
        """Converts a sys.exc_info()-style tuple of values into a string."""
        exctype, value, tb = err
        # Skip test runner traceback levels
        while tb and self._is_relevant_tb_level(tb):
            tb = tb.tb_next
        if exctype is test.failureException:
            # Skip assert*() traceback levels
            length = self._count_relevant_tb_levels(tb)
            return ''.join(traceback.format_exception(exctype, value, tb, length))
        return ''.join(traceback.format_exception(exctype, value, tb))

    def _is_relevant_tb_level(self, tb):
        return tb.tb_frame.f_globals.has_key('__unittest')

    def _count_relevant_tb_levels(self, tb):
        length = 0
        while tb and not self._is_relevant_tb_level(tb):
            length += 1
            tb = tb.tb_next
        return length

    def __repr__(self):
        return "<%s run=%i errors=%i failures=%i>" % \
               (_strclass(self.__class__), self.testsRun, len(self.errors),
                len(self.failures))

class TestCase:
    """A class whose instances are single test cases.

    By default, the test code itself should be placed in a method named
    'runTest'.

    If the fixture may be used for many test cases, create as
    many test methods as are needed. When instantiating such a TestCase
    subclass, specify in the constructor arguments the name of the test method
    that the instance is to execute.

    Test authors should subclass TestCase for their own tests. Construction
    and deconstruction of the test's environment ('fixture') can be
    implemented by overriding the 'setUp' and 'tearDown' methods respectively.

    If it is necessary to override the __init__ method, the base class
    __init__ method must always be called. It is important that subclasses
    should not change the signature of their __init__ method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.
    """

    # This attribute determines which exception will be raised when
    # the instance's assertion methods fail; test methods raising this
    # exception will be deemed to have 'failed' rather than 'errored'

    failureException = AssertionError

    def __init__(self, methodName='runTest'):
        """Create an instance of the class that will use the named test
           method when executed. Raises a ValueError if the instance does
           not have a method with the specified name.
        """
        try:
            self._testMethodName = methodName
            testMethod = getattr(self, methodName)
            self._testMethodDoc = testMethod.__doc__
        except AttributeError:
            raise ValueError, "no such test method in %s: %s" % \
                  (self.__class__, methodName)

    def setUp(self):
        "Hook method for setting up the test fixture before exercising it."
        pass

    def tearDown(self):
        "Hook method for deconstructing the test fixture after testing it."
        pass

    def countTestCases(self):
        return 1

    def defaultTestResult(self):
        return TestResult()

    def shortDescription(self):
        """Returns a one-line description of the test, or None if no
        description has been provided.

        The default implementation of this method returns the first line of
        the specified test method's docstring.
        """
        doc = self._testMethodDoc
        return doc and doc.split("\n")[0].strip() or None

    def id(self):
        return "%s.%s" % (_strclass(self.__class__), self._testMethodName)

    def __str__(self):
        return "%s (%s)" % (self._testMethodName, _strclass(self.__class__))

    def __repr__(self):
        return "<%s testMethod=%s>" % \
               (_strclass(self.__class__), self._testMethodName)

    def run(self, result=None):
        if result is None: result = self.defaultTestResult()
        result.startTest(self)
        testMethod = getattr(self, self._testMethodName)
        try:
            try:
                self.setUp()
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())
                return

            ok = False
            try:
                testMethod()
                ok = True
            except self.failureException:
                result.addFailure(self, self._exc_info())
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())

            try:
                self.tearDown()
            except KeyboardInterrupt:
                raise
            except:
                result.addError(self, self._exc_info())
                ok = False
            if ok: result.addSuccess(self)
        finally:
            result.stopTest(self)

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def debug(self):
        """Run the test without collecting errors in a TestResult"""
        self.setUp()
        getattr(self, self._testMethodName)()
        self.tearDown()

    def _exc_info(self):
        """Return a version of sys.exc_info() with the traceback frame
           minimised; usually the top level of the traceback frame is not
           needed.
        """
        exctype, excvalue, tb = sys.exc_info()
        if sys.platform[:4] == 'java': ## tracebacks look different in Jython
            return (exctype, excvalue, tb)
        return (exctype, excvalue, tb)

    def fail(self, msg=None):
        """Fail immediately, with the given message."""
        raise self.failureException, msg

    def failIf(self, expr, msg=None):
        "Fail the test if the expression is true."
        if expr: raise self.failureException, msg

    def failUnless(self, expr, msg=None):
        """Fail the test unless the expression is true."""
        if not expr: raise self.failureException, msg

    def failUnlessRaises(self, excClass, callableObj, *args, **kwargs):
        """Fail unless an exception of class excClass is thrown
           by callableObj when invoked with arguments args and keyword
           arguments kwargs. If a different type of exception is
           thrown, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.
        """
        try:
            callableObj(*args, **kwargs)
        except excClass:
            return
        else:
            if hasattr(excClass,'__name__'): excName = excClass.__name__
            else: excName = str(excClass)
            raise self.failureException, "%s not raised" % excName

    def failUnlessEqual(self, first, second, msg=None):
        """Fail if the two objects are unequal as determined by the '=='
           operator.
        """
        if not first == second:
            raise self.failureException, \
                  (msg or '%r != %r' % (first, second))

    def failIfEqual(self, first, second, msg=None):
        """Fail if the two objects are equal as determined by the '=='
           operator.
        """
        if first == second:
            raise self.failureException, \
                  (msg or '%r == %r' % (first, second))

    def failUnlessAlmostEqual(self, first, second, places=7, msg=None):
        """Fail if the two objects are unequal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most signficant digit).
        """
        if round(second-first, places) != 0:
            raise self.failureException, \
                  (msg or '%r != %r within %r places' % (first, second, places))

    def failIfAlmostEqual(self, first, second, places=7, msg=None):
        """Fail if the two objects are equal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most signficant digit).
        """
        if round(second-first, places) == 0:
            raise self.failureException, \
                  (msg or '%r == %r within %r places' % (first, second, places))

    # Synonyms for assertion methods

    assertEqual = assertEquals = failUnlessEqual

    assertNotEqual = assertNotEquals = failIfEqual

    assertAlmostEqual = assertAlmostEquals = failUnlessAlmostEqual

    assertNotAlmostEqual = assertNotAlmostEquals = failIfAlmostEqual

    assertRaises = failUnlessRaises

    assert_ = assertTrue = failUnless

    assertFalse = failIf



class TestSuite:
    """A test suite is a composite test consisting of a number of TestCases.

    For use, create an instance of TestSuite, then add test case instances.
    When all tests have been added, the suite can be passed to a test
    runner, such as TextTestRunner. It will run the individual test cases
    in the order in which they were added, aggregating the results. When
    subclassing, do not forget to call the base class constructor.
    """
    def __init__(self, tests=()):
        self._tests = []
        self.addTests(tests)

    def __repr__(self):
        return "<%s tests=%s>" % (_strclass(self.__class__), self._tests)

    __str__ = __repr__

    def __iter__(self):
        return iter(self._tests)

    def countTestCases(self):
        cases = 0
        for test in self._tests:
            cases += test.countTestCases()
        return cases

    def addTest(self, test):
        # sanity checks
        if not callable(test):
            raise TypeError("the test to add must be callable")
        if (isinstance(test, (type, types.ClassType)) and
            issubclass(test, (TestCase, TestSuite))):
            raise TypeError("TestCases and TestSuites must be instantiated "
                            "before passing them to addTest()")
        self._tests.append(test)

    def addTests(self, tests):
        if isinstance(tests, basestring):
            raise TypeError("tests must be an iterable of tests, not a string")
        for test in tests:
            self.addTest(test)

    def run(self, result):
        for test in self._tests:
            if result.shouldStop:
                break
            test(result)
        return result

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def debug(self):
        """Run the tests without collecting errors in a TestResult"""
        for test in self._tests: test.debug()


class FunctionTestCase(TestCase):
    """A test case that wraps a test function.

    This is useful for slipping pre-existing test functions into the
    PyUnit framework. Optionally, set-up and tidy-up functions can be
    supplied. As with TestCase, the tidy-up ('tearDown') function will
    always be called if the set-up ('setUp') function ran successfully.
    """

    def __init__(self, testFunc, setUp=None, tearDown=None,
                 description=None):
        TestCase.__init__(self)
        self.__setUpFunc = setUp
        self.__tearDownFunc = tearDown
        self.__testFunc = testFunc
        self.__description = description

    def setUp(self):
        if self.__setUpFunc is not None:
            self.__setUpFunc()

    def tearDown(self):
        if self.__tearDownFunc is not None:
            self.__tearDownFunc()

    def runTest(self):
        self.__testFunc()

    def id(self):
        return self.__testFunc.__name__

    def __str__(self):
        return "%s (%s)" % (_strclass(self.__class__), self.__testFunc.__name__)

    def __repr__(self):
        return "<%s testFunc=%s>" % (_strclass(self.__class__), self.__testFunc)

    def shortDescription(self):
        if self.__description is not None: return self.__description
        doc = self.__testFunc.__doc__
        return doc and doc.split("\n")[0].strip() or None



##############################################################################
# Locating and loading tests
##############################################################################

class TestLoader:
    """This class is responsible for loading tests according to various
    criteria and returning them wrapped in a Test
    """
    testMethodPrefix = 'test'
    sortTestMethodsUsing = cmp
    suiteClass = TestSuite

    def loadTestsFromTestCase(self, testCaseClass):
        """Return a suite of all tests cases contained in testCaseClass"""
        if issubclass(testCaseClass, TestSuite):
            raise TypeError("Test cases should not be derived from TestSuite. Maybe you meant to derive from TestCase?")
        testCaseNames = self.getTestCaseNames(testCaseClass)
        if not testCaseNames and hasattr(testCaseClass, 'runTest'):
            testCaseNames = ['runTest']
        return self.suiteClass(map(testCaseClass, testCaseNames))

    def loadTestsFromModule(self, module):
        """Return a suite of all tests cases contained in the given module"""
        tests = []
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, (type, types.ClassType)) and
                issubclass(obj, TestCase)):
                tests.append(self.loadTestsFromTestCase(obj))
        return self.suiteClass(tests)

    def loadTestsFromName(self, name, module=None):
        """Return a suite of all tests cases given a string specifier.

        The name may resolve either to a module, a test case class, a
        test method within a test case class, or a callable object which
        returns a TestCase or TestSuite instance.

        The method optionally resolves the names relative to a given module.
        """
        parts = name.split('.')
        if module is None:
            parts_copy = parts[:]
            while parts_copy:
                try:
                    module = __import__('.'.join(parts_copy))
                    break
                except ImportError:
                    del parts_copy[-1]
                    if not parts_copy: raise
            parts = parts[1:]
        obj = module
        for part in parts:
            parent, obj = obj, getattr(obj, part)

        if type(obj) == types.ModuleType:
            return self.loadTestsFromModule(obj)
        elif (isinstance(obj, (type, types.ClassType)) and
              issubclass(obj, TestCase)):
            return self.loadTestsFromTestCase(obj)
        elif type(obj) == types.UnboundMethodType:
            return parent(obj.__name__)
        elif isinstance(obj, TestSuite):
            return obj
        elif callable(obj):
            test = obj()
            if not isinstance(test, (TestCase, TestSuite)):
                raise ValueError, \
                      "calling %s returned %s, not a test" % (obj,test)
            return test
        else:
            raise ValueError, "don't know how to make test from: %s" % obj

    def loadTestsFromNames(self, names, module=None):
        """Return a suite of all tests cases found using the given sequence
        of string specifiers. See 'loadTestsFromName()'.
        """
        suites = [self.loadTestsFromName(name, module) for name in names]
        return self.suiteClass(suites)

    def getTestCaseNames(self, testCaseClass):
        """Return a sorted sequence of method names found within testCaseClass
        """
        def isTestMethod(attrname, testCaseClass=testCaseClass, prefix=self.testMethodPrefix):
            return attrname.startswith(prefix) and callable(getattr(testCaseClass, attrname))
        testFnNames = filter(isTestMethod, dir(testCaseClass))
        for baseclass in testCaseClass.__bases__:
            for testFnName in self.getTestCaseNames(baseclass):
                if testFnName not in testFnNames:  # handle overridden methods
                    testFnNames.append(testFnName)
        if self.sortTestMethodsUsing:
            testFnNames.sort(self.sortTestMethodsUsing)
        return testFnNames



defaultTestLoader = TestLoader()


##############################################################################
# Patches for old functions: these functions should be considered obsolete
##############################################################################

def _makeLoader(prefix, sortUsing, suiteClass=None):
    loader = TestLoader()
    loader.sortTestMethodsUsing = sortUsing
    loader.testMethodPrefix = prefix
    if suiteClass: loader.suiteClass = suiteClass
    return loader

def getTestCaseNames(testCaseClass, prefix, sortUsing=cmp):
    return _makeLoader(prefix, sortUsing).getTestCaseNames(testCaseClass)

def makeSuite(testCaseClass, prefix='test', sortUsing=cmp, suiteClass=TestSuite):
    return _makeLoader(prefix, sortUsing, suiteClass).loadTestsFromTestCase(testCaseClass)

def findTestCases(module, prefix='test', sortUsing=cmp, suiteClass=TestSuite):
    return _makeLoader(prefix, sortUsing, suiteClass).loadTestsFromModule(module)


##############################################################################
# Text UI
##############################################################################

class _WritelnDecorator:
    """Used to decorate file-like objects with a handy 'writeln' method"""
    def __init__(self,stream):
        self.stream = stream

    def __getattr__(self, attr):
        return getattr(self.stream,attr)

    def writeln(self, arg=None):
        if arg: self.write(arg)
        self.write('\n') # text-mode streams translate to \r\n if needed


class _TextTestResult(TestResult):
    """A test result class that can print formatted text results to a stream.

    Used by TextTestRunner.
    """
    separator1 = '=' * 70
    separator2 = '-' * 70

    def __init__(self, stream, descriptions, verbosity):
        TestResult.__init__(self)
        self.stream = stream
        self.showAll = verbosity > 1
        self.dots = verbosity == 1
        self.descriptions = descriptions

    def getDescription(self, test):
        if self.descriptions:
            return test.shortDescription() or str(test)
        else:
            return str(test)

    def startTest(self, test):
        TestResult.startTest(self, test)
        if self.showAll:
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")

    def addSuccess(self, test):
        TestResult.addSuccess(self, test)
        if self.showAll:
            self.stream.writeln("ok")
        elif self.dots:
            self.stream.write('.')

    def addError(self, test, err):
        TestResult.addError(self, test, err)
        if self.showAll:
            self.stream.writeln("ERROR")
        elif self.dots:
            self.stream.write('E')

    def addFailure(self, test, err):
        TestResult.addFailure(self, test, err)
        if self.showAll:
            self.stream.writeln("FAIL")
        elif self.dots:
            self.stream.write('F')

    def printErrors(self):
        if self.dots or self.showAll:
            self.stream.writeln()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln("%s: %s" % (flavour,self.getDescription(test)))
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)


class TextTestRunner:
    """A test runner class that displays results in textual form.

    It prints out the names of tests as they are run, errors as they
    occur, and a summary of the results at the end of the test run.
    """
    def __init__(self, stream=sys.stderr, descriptions=1, verbosity=1):
        self.stream = _WritelnDecorator(stream)
        self.descriptions = descriptions
        self.verbosity = verbosity

    def _makeResult(self):
        return _TextTestResult(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        "Run the given test case or test suite."
        result = self._makeResult()
        startTime = time.time()
        test(result)
        stopTime = time.time()
        timeTaken = stopTime - startTime
        result.printErrors()
        self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()
        if not result.wasSuccessful():
            self.stream.write("FAILED (")
            failed, errored = map(len, (result.failures, result.errors))
            if failed:
                self.stream.write("failures=%d" % failed)
            if errored:
                if failed: self.stream.write(", ")
                self.stream.write("errors=%d" % errored)
            self.stream.writeln(")")
        else:
            self.stream.writeln("OK")
        return result



##############################################################################
# Facilities for running tests from the command line
##############################################################################

class TestProgram:
    """A command-line program that runs a set of tests; this is primarily
       for making test modules conveniently executable.
    """
    USAGE = """\
Usage: %(progName)s [options] [test] [...]

Options:
  -h, --help       Show this message
  -v, --verbose    Verbose output
  -q, --quiet      Minimal output

Examples:
  %(progName)s                               - run default set of tests
  %(progName)s MyTestSuite                   - run suite 'MyTestSuite'
  %(progName)s MyTestCase.testSomething      - run MyTestCase.testSomething
  %(progName)s MyTestCase                    - run all 'test*' test methods
                                               in MyTestCase
"""
    def __init__(self, module='__main__', defaultTest=None,
                 argv=None, testRunner=None, testLoader=defaultTestLoader):
        if type(module) == type(''):
            self.module = __import__(module)
            for part in module.split('.')[1:]:
                self.module = getattr(self.module, part)
        else:
            self.module = module
        if argv is None:
            argv = sys.argv
        self.verbosity = 1
        self.defaultTest = defaultTest
        self.testRunner = testRunner
        self.testLoader = testLoader
        self.progName = os.path.basename(argv[0])
        self.parseArgs(argv)
        self.runTests()

    def usageExit(self, msg=None):
        if msg: print msg
        print self.USAGE % self.__dict__
        sys.exit(2)

    def parseArgs(self, argv):
        import getopt
        try:
            options, args = getopt.getopt(argv[1:], 'hHvq',
                                          ['help','verbose','quiet'])
            for opt, value in options:
                if opt in ('-h','-H','--help'):
                    self.usageExit()
                if opt in ('-q','--quiet'):
                    self.verbosity = 0
                if opt in ('-v','--verbose'):
                    self.verbosity = 2
            if len(args) == 0 and self.defaultTest is None:
                self.test = self.testLoader.loadTestsFromModule(self.module)
                return
            if len(args) > 0:
                self.testNames = args
            else:
                self.testNames = (self.defaultTest,)
            self.createTests()
        except getopt.error, msg:
            self.usageExit(msg)

    def createTests(self):
        self.test = self.testLoader.loadTestsFromNames(self.testNames,
                                                       self.module)

    def runTests(self):
        if self.testRunner is None:
            self.testRunner = TextTestRunner(verbosity=self.verbosity)
        result = self.testRunner.run(self.test)
        sys.exit(not result.wasSuccessful())

main = TestProgram


##############################################################################
# Executing this module from the command line
##############################################################################

if __name__ == "__main__":
    main(module=None)
