"""
@author: Kun Han
"""

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='memfnet',
        version='0.0.1',
        description='A package for learning representationos of battery voltage profiles',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='',
        author=['Kun Han'],
        author_email=['2172811165@qq.com'],
        license='MIT',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=[],
        extras_require={},
        classifiers=[],
        test_suite='',
        tests_require=[],
        scripts=[]
    )
