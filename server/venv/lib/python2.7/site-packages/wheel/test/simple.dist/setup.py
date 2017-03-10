from setuptools import setup

try:
    unicode
    def u8(s):
        return s.decode('unicode-escape').encode('utf-8')
except NameError:
    def u8(s):
        return s.encode('utf-8')

setup(name='simple.dist',
      version='0.1',
      description=u8('A testing distribution \N{SNOWMAN}'),
      packages=['simpledist'],
      extras_require={'voting': ['beaglevote']},
      )

