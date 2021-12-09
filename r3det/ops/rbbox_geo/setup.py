from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rbbox_geo_cuda',
    ext_modules=[
        CUDAExtension('rbbox_geo_cuda',
                      ['src/rbbox_geo_cuda.cpp', 'src/rbbox_geo_kernel.cu']),
    ],
    cmdclass={'build_ext': BuildExtension})
