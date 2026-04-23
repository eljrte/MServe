"""
编译CUDA UVA扩展

运行:
    python setup_cuda_uva.py install
或者:
    python setup_cuda_uva.py build_ext --inplace

注意：需要CUDA toolkit和nvcc编译器
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_uva',
    ext_modules=[
        CUDAExtension(
            name='cuda_uva',
            sources=['cuda_uva_kernel.cpp'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 30系列
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 40系列
                    '--use_fast_math',
                ],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
