import os
import wheel.install
import wheel.archive
import hashlib
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO
import codecs
import zipfile
import pytest
import shutil
import tempfile
from contextlib import contextmanager

@contextmanager
def environ(key, value):
    old_value = os.environ.get(key)
    try:
        os.environ[key] = value
        yield
    finally:
        if old_value is None:
            del os.environ[key]
        else:
            os.environ[key] = old_value

@contextmanager
def temporary_directory():
    # tempfile.TemporaryDirectory doesn't exist in Python 2.
    tempdir = tempfile.mkdtemp()
    try:
        yield tempdir
    finally:
        shutil.rmtree(tempdir)

@contextmanager
def readable_zipfile(path):
    # zipfile.ZipFile() isn't a context manager under Python 2.
    zf = zipfile.ZipFile(path, 'r')
    try:
        yield zf
    finally:
        zf.close()


def test_verifying_zipfile():
    if not hasattr(zipfile.ZipExtFile, '_update_crc'):
        pytest.skip('No ZIP verification. Missing ZipExtFile._update_crc.')
    
    sio = StringIO()
    zf = zipfile.ZipFile(sio, 'w')
    zf.writestr("one", b"first file")
    zf.writestr("two", b"second file")
    zf.writestr("three", b"third file")
    zf.close()
    
    # In default mode, VerifyingZipFile checks the hash of any read file
    # mentioned with set_expected_hash(). Files not mentioned with
    # set_expected_hash() are not checked.
    vzf = wheel.install.VerifyingZipFile(sio, 'r')
    vzf.set_expected_hash("one", hashlib.sha256(b"first file").digest())
    vzf.set_expected_hash("three", "blurble")
    vzf.open("one").read()
    vzf.open("two").read()
    try:
        vzf.open("three").read()
    except wheel.install.BadWheelFile:
        pass
    else:
        raise Exception("expected exception 'BadWheelFile()'")
    
    # In strict mode, VerifyingZipFile requires every read file to be
    # mentioned with set_expected_hash().
    vzf.strict = True
    try:
        vzf.open("two").read()
    except wheel.install.BadWheelFile:
        pass
    else:
        raise Exception("expected exception 'BadWheelFile()'")
        
    vzf.set_expected_hash("two", None)
    vzf.open("two").read()
    
def test_pop_zipfile():
    sio = StringIO()
    zf = wheel.install.VerifyingZipFile(sio, 'w')
    zf.writestr("one", b"first file")
    zf.writestr("two", b"second file")
    zf.close()
    
    try:
        zf.pop()
    except RuntimeError:
        pass # already closed
    else:
        raise Exception("expected RuntimeError")
    
    zf = wheel.install.VerifyingZipFile(sio, 'a')
    zf.pop()
    zf.close()
    
    zf = wheel.install.VerifyingZipFile(sio, 'r')
    assert len(zf.infolist()) == 1

def test_zipfile_timestamp():
    # An environment variable can be used to influence the timestamp on
    # TarInfo objects inside the zip.  See issue #143.  TemporaryDirectory is
    # not a context manager under Python 3.
    with temporary_directory() as tempdir:
        for filename in ('one', 'two', 'three'):
            path = os.path.join(tempdir, filename)
            with codecs.open(path, 'w', encoding='utf-8') as fp:
                fp.write(filename + '\n')
        zip_base_name = os.path.join(tempdir, 'dummy')
        # The earliest date representable in TarInfos, 1980-01-01
        with environ('SOURCE_DATE_EPOCH', '315576060'):
            zip_filename = wheel.archive.make_wheelfile_inner(
                zip_base_name, tempdir)
        with readable_zipfile(zip_filename) as zf:
            for info in zf.infolist():
                assert info.date_time[:3] == (1980, 1, 1)

def test_zipfile_attributes():
    # With the change from ZipFile.write() to .writestr(), we need to manually
    # set member attributes.
    with temporary_directory() as tempdir:
        files = (('foo', 0o644), ('bar', 0o755))
        for filename, mode in files:
            path = os.path.join(tempdir, filename)
            with codecs.open(path, 'w', encoding='utf-8') as fp:
                fp.write(filename + '\n')
            os.chmod(path, mode)
        zip_base_name = os.path.join(tempdir, 'dummy')
        zip_filename = wheel.archive.make_wheelfile_inner(
            zip_base_name, tempdir)
        with readable_zipfile(zip_filename) as zf:
            for filename, mode in files:
                info = zf.getinfo(os.path.join(tempdir, filename))
                assert info.external_attr == (mode | 0o100000) << 16
                assert info.compress_type == zipfile.ZIP_DEFLATED
