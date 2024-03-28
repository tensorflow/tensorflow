import os
import subprocess
import contextlib
import functools
import tempfile
import shutil
import operator


@contextlib.contextmanager
def pushd(dir):
    orig = os.getcwd()
    os.chdir(dir)
    try:
        yield dir
    finally:
        os.chdir(orig)


@contextlib.contextmanager
def tarball_context(url, target_dir=None, runner=None, pushd=pushd):
    """
    Get a tarball, extract it, change to that directory, yield, then
    clean up.
    `runner` is the function to invoke commands.
    `pushd` is a context manager for changing the directory.
    """
    if target_dir is None:
        target_dir = os.path.basename(url).replace('.tar.gz', '').replace('.tgz', '')
    if runner is None:
        runner = functools.partial(subprocess.check_call, shell=True)
    # In the tar command, use --strip-components=1 to strip the first path and
    #  then
    #  use -C to cause the files to be extracted to {target_dir}. This ensures
    #  that we always know where the files were extracted.
    runner('mkdir {target_dir}'.format(**vars()))
    try:
        getter = 'wget {url} -O -'
        extract = 'tar x{compression} --strip-components=1 -C {target_dir}'
        cmd = ' | '.join((getter, extract))
        runner(cmd.format(compression=infer_compression(url), **vars()))
        with pushd(target_dir):
            yield target_dir
    finally:
        runner('rm -Rf {target_dir}'.format(**vars()))


def infer_compression(url):
    """
    Given a URL or filename, infer the compression code for tar.
    """
    # cheat and just assume it's the last two characters
    compression_indicator = url[-2:]
    mapping = dict(gz='z', bz='j', xz='J')
    # Assume 'z' (gzip) if no match
    return mapping.get(compression_indicator, 'z')


@contextlib.contextmanager
def temp_dir(remover=shutil.rmtree):
    """
    Create a temporary directory context. Pass a custom remover
    to override the removal behavior.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        remover(temp_dir)


@contextlib.contextmanager
def repo_context(url, branch=None, quiet=True, dest_ctx=temp_dir):
    """
    Check out the repo indicated by url.

    If dest_ctx is supplied, it should be a context manager
    to yield the target directory for the check out.
    """
    exe = 'git' if 'git' in url else 'hg'
    with dest_ctx() as repo_dir:
        cmd = [exe, 'clone', url, repo_dir]
        if branch:
            cmd.extend(['--branch', branch])
        devnull = open(os.path.devnull, 'w')
        stdout = devnull if quiet else None
        subprocess.check_call(cmd, stdout=stdout)
        yield repo_dir


@contextlib.contextmanager
def null():
    yield


class ExceptionTrap:
    """
    A context manager that will catch certain exceptions and provide an
    indication they occurred.

    >>> with ExceptionTrap() as trap:
    ...     raise Exception()
    >>> bool(trap)
    True

    >>> with ExceptionTrap() as trap:
    ...     pass
    >>> bool(trap)
    False

    >>> with ExceptionTrap(ValueError) as trap:
    ...     raise ValueError("1 + 1 is not 3")
    >>> bool(trap)
    True

    >>> with ExceptionTrap(ValueError) as trap:
    ...     raise Exception()
    Traceback (most recent call last):
    ...
    Exception

    >>> bool(trap)
    False
    """

    exc_info = None, None, None

    def __init__(self, exceptions=(Exception,)):
        self.exceptions = exceptions

    def __enter__(self):
        return self

    @property
    def type(self):
        return self.exc_info[0]

    @property
    def value(self):
        return self.exc_info[1]

    @property
    def tb(self):
        return self.exc_info[2]

    def __exit__(self, *exc_info):
        type = exc_info[0]
        matches = type and issubclass(type, self.exceptions)
        if matches:
            self.exc_info = exc_info
        return matches

    def __bool__(self):
        return bool(self.type)

    def raises(self, func, *, _test=bool):
        """
        Wrap func and replace the result with the truth
        value of the trap (True if an exception occurred).

        First, give the decorator an alias to support Python 3.8
        Syntax.

        >>> raises = ExceptionTrap(ValueError).raises

        Now decorate a function that always fails.

        >>> @raises
        ... def fail():
        ...     raise ValueError('failed')
        >>> fail()
        True
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with ExceptionTrap(self.exceptions) as trap:
                func(*args, **kwargs)
            return _test(trap)

        return wrapper

    def passes(self, func):
        """
        Wrap func and replace the result with the truth
        value of the trap (True if no exception).

        First, give the decorator an alias to support Python 3.8
        Syntax.

        >>> passes = ExceptionTrap(ValueError).passes

        Now decorate a function that always fails.

        >>> @passes
        ... def fail():
        ...     raise ValueError('failed')

        >>> fail()
        False
        """
        return self.raises(func, _test=operator.not_)


class suppress(contextlib.suppress, contextlib.ContextDecorator):
    """
    A version of contextlib.suppress with decorator support.

    >>> @suppress(KeyError)
    ... def key_error():
    ...     {}['']
    >>> key_error()
    """
