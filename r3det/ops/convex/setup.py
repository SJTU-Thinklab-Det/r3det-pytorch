from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='convex_ext',
    ext_modules=[
        CUDAExtension(
            'convex_ext',
            ['src/convex_cpu.cpp', 'src/convex_cuda.cu', 'src/convex_ext.cpp'
             ]),
    ],
    cmdclass={'build_ext': BuildExtension})
