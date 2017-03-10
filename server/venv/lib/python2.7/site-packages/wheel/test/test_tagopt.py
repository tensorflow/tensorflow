"""
Tests for the bdist_wheel tag options (--python-tag, --universal, and
--plat-name)
"""

import sys
import shutil
import pytest
import py.path
import tempfile
import subprocess

SETUP_PY = """\
from setuptools import setup, Extension

setup(
    name="Test",
    version="1.0",
    author_email="author@example.com",
    py_modules=["test"],
    {ext_modules}
)
"""

EXT_MODULES = "ext_modules=[Extension('_test', sources=['test.c'])],"

@pytest.fixture
def temp_pkg(request, ext=False):
    tempdir = tempfile.mkdtemp()
    def fin():
        shutil.rmtree(tempdir)
    request.addfinalizer(fin)
    temppath = py.path.local(tempdir)
    temppath.join('test.py').write('print("Hello, world")')
    if ext:
        temppath.join('test.c').write('#include <stdio.h>')
        setup_py = SETUP_PY.format(ext_modules=EXT_MODULES)
    else:
        setup_py = SETUP_PY.format(ext_modules='')
    temppath.join('setup.py').write(setup_py)
    return temppath

@pytest.fixture
def temp_ext_pkg(request):
    return temp_pkg(request, ext=True)

def test_default_tag(temp_pkg):
    subprocess.check_call([sys.executable, 'setup.py', 'bdist_wheel'],
            cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename == 'Test-1.0-py%s-none-any.whl' % (sys.version[0],)
    assert wheels[0].ext == '.whl'

def test_explicit_tag(temp_pkg):
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel', '--python-tag=py32'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.startswith('Test-1.0-py32-')
    assert wheels[0].ext == '.whl'

def test_universal_tag(temp_pkg):
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel', '--universal'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.startswith('Test-1.0-py2.py3-')
    assert wheels[0].ext == '.whl'

def test_universal_beats_explicit_tag(temp_pkg):
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel', '--universal', '--python-tag=py32'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.startswith('Test-1.0-py2.py3-')
    assert wheels[0].ext == '.whl'

def test_universal_in_setup_cfg(temp_pkg):
    temp_pkg.join('setup.cfg').write('[bdist_wheel]\nuniversal=1')
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.startswith('Test-1.0-py2.py3-')
    assert wheels[0].ext == '.whl'

def test_pythontag_in_setup_cfg(temp_pkg):
    temp_pkg.join('setup.cfg').write('[bdist_wheel]\npython_tag=py32')
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.startswith('Test-1.0-py32-')
    assert wheels[0].ext == '.whl'

def test_legacy_wheel_section_in_setup_cfg(temp_pkg):
    temp_pkg.join('setup.cfg').write('[wheel]\nuniversal=1')
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.startswith('Test-1.0-py2.py3-')
    assert wheels[0].ext == '.whl'

def test_plat_name_purepy(temp_pkg):
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel', '--plat-name=testplat.pure'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.endswith('-testplat_pure.whl')
    assert wheels[0].ext == '.whl'

def test_plat_name_ext(temp_ext_pkg):
    try:
        subprocess.check_call(
            [sys.executable, 'setup.py', 'bdist_wheel', '--plat-name=testplat.arch'],
            cwd=str(temp_ext_pkg))
    except subprocess.CalledProcessError:
        pytest.skip("Cannot compile C Extensions")
    dist_dir = temp_ext_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.endswith('-testplat_arch.whl')
    assert wheels[0].ext == '.whl'

def test_plat_name_purepy_in_setupcfg(temp_pkg):
    temp_pkg.join('setup.cfg').write('[bdist_wheel]\nplat_name=testplat.pure')
    subprocess.check_call(
        [sys.executable, 'setup.py', 'bdist_wheel'],
        cwd=str(temp_pkg))
    dist_dir = temp_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.endswith('-testplat_pure.whl')
    assert wheels[0].ext == '.whl'

def test_plat_name_ext_in_setupcfg(temp_ext_pkg):
    temp_ext_pkg.join('setup.cfg').write('[bdist_wheel]\nplat_name=testplat.arch')
    try:
        subprocess.check_call(
            [sys.executable, 'setup.py', 'bdist_wheel'],
            cwd=str(temp_ext_pkg))
    except subprocess.CalledProcessError:
        pytest.skip("Cannot compile C Extensions")
    dist_dir = temp_ext_pkg.join('dist')
    assert dist_dir.check(dir=1)
    wheels = dist_dir.listdir()
    assert len(wheels) == 1
    assert wheels[0].basename.endswith('-testplat_arch.whl')
    assert wheels[0].ext == '.whl'
