from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_uva',
    ext_modules=[
        CUDAExtension(
            name='cuda_uva',
            sources=['cuda_uva_kernel_v2.cpp'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'],
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
