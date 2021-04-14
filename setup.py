from os import path

from setuptools import setup

FULLVERSION = '0.0.1'
VERSION = FULLVERSION

REQUIREMENTS_PATH = path.join(path.dirname(path.realpath(__file__)), "requirements.txt")

if path.isfile(REQUIREMENTS_PATH):
    with open(REQUIREMENTS_PATH) as infile:
        INSTALL_REQUIRES = infile.read().splitlines()

setup(name='xdem',
      version=FULLVERSION,
      description='',
      url='',
      author='The GlacioHack Team',
      license='BSD-3',
      packages=['xdem'],
      install_requires=INSTALL_REQUIRES,
      extras_require={'rioxarray': ['rioxarray'], 'richdem': ['richdem'], 'pdal': [
          'pdal'], 'opencv': ['opencv'], "pytransform3d": ["pytransform3d"]},
      scripts=[],
      zip_safe=False)

write_version = True


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = path.join(path.dirname(__file__), 'xdem',
                             'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()
