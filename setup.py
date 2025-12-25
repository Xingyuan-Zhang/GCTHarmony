from setuptools import setup, find_packages

setup(
    name='GCTHarmony',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'owlready2',
        'numpy',
        'pandas',
        'umap-learn',
        'matplotlib',
        'rdflib',
        'scipy',
        'openai'
    ],
    include_package_data=True,
    package_data={                          
        "GCTHarmony": ["data/*"]
    },
    author='Xingyuan Zhang',
    description='LLM-based Cell Type Annotation Harmonization Tool',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Xingyuan-Zhang/GCTHarmony',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.12',
)
