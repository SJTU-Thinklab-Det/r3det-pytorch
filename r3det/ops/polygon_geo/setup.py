from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='polygon_geo_cpu',
    ext_modules=[
        CUDAExtension('polygon_geo_cpu', ['src/polygon_geo_cpu.cpp']),
    ],
    cmdclass={'build_ext': BuildExtension})
