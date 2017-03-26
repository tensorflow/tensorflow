import os
import sys
import pip
import glob
import shutil

here = os.path.abspath(os.path.dirname(__file__))

def usage():
    print("Usage: re-vendor.py [clean|vendor]")
    sys.exit(1)

def clean():
    for fn in os.listdir(here):
        dirname = os.path.join(here, fn)
        if os.path.isdir(dirname):
            shutil.rmtree(dirname)
    # six is a single file, not a package
    os.unlink(os.path.join(here, 'six.py'))

def vendor():
    pip.main(['install', '-t', here, '-r', 'vendor.txt'])
    for dirname in glob.glob('*.egg-info'):
        shutil.rmtree(dirname)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
    if sys.argv[1] == 'clean':
        clean()
    elif sys.argv[1] == 'vendor':
        vendor()
    else:
        usage()
