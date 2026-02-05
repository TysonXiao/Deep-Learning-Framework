from setuptools import setup, find_packages

setup(
    name='dezero',             # 包的名字
    version='0.1.0',           # 版本号
    packages=find_packages(),  # 自动寻找项目中的所有包（带 __init__.py 的文件夹）
    install_requires=[         # 列出你的框架依赖的库
        'numpy',
    ],
    author='Tyson',
    description='A simple deep learning framework',
)