from setuptools import setup, find_packages

setup(
    name='tokeniser-py',
    version='0.1.4',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'tokeniser': ['data/*.json'],
    },
    install_requires=[
        'numpy',
        'torch',
        'regex',
        'tqdm'
    ],
    long_description=open('README.md', encoding='utf-8').read() + '\n\n' + open('CHANGELOG.txt', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Tasmay Pankaj Tibrewal',
    author_email='keshavtibrewal2@gmail.com',
    url='https://github.com/Tasmay-Tibrewal/tokeniser-py',
    license='MIT',
    keywords=['Tokens', 'Tokeniser', 'Tokenizer', 'LLMs', 'LMs', 'LLM', 'LM', 'Language Model', 'Language Models', 'Large Language Models', 'Large Language Model'],
    description='A custom tokeniser with a 131,072-token vocabulary derived from 0.5B (val) and 1B (val+test) tokens in SlimPajama. Uses a novel token generation algorithm and a dynamic programming-based segmentation method for fast, interpretable tokenisation, which can also be used for tokeniation on custom token maps.',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
