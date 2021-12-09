from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rnms_ext',
    ext_modules=[
        CUDAExtension('rnms_ext', [
            'src/rnms_ext.cpp',
            'src/rcpu/rnms_cpu.cpp',
            'src/rcuda/rnms_cuda.cpp',
            'src/rcuda/rnms_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
