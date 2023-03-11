from setuptools import setup, Extension
import numpy
'''
module1 = Extension('prange_test', sources=['prange_test.pyx'],
                                            include_dirs=[numpy.get_include()],
                    extra_compile_args = ['-fopenmp'], extra_link_args = ['-fopenmp']
)
'''
module2 = Extension('nppar', sources=['create2Dpar.pyx'],
                                            include_dirs=[numpy.get_include()],
                    extra_compile_args = ['-fopenmp'], extra_link_args = ['-fopenmp']
)


setup(
    name='prange_tools',
    version='0.1',
    #packages=[''],
    #url='',
    #license='',
    author='User',
    #author_email='',
    #description='',
    ext_modules=[module2]
)
