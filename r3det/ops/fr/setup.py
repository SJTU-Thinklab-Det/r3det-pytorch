from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='feature_refine_cuda',
    ext_modules=[
        CUDAExtension('feature_refine_cuda', [
            'src/feature_refine_cuda.cpp',
            'src/feature_refine_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
