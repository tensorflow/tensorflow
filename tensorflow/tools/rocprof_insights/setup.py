from setuptools import setup, find_packages

setup(
    name='rocprof-insights',
    version='0.1.0',
    author='cj401-amd',
    author_email='chunyjin@amd.com',
    description='A package to perform EDA on rocprof data (v1, v2, v3).',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'plotly',
        'kaleido',
    ],
    entry_points={
        'console_scripts': [
            'rocprofiler-insights=rocprof_insights.cli:main'
        ]
    },
    python_requires='>=3.7',
)
