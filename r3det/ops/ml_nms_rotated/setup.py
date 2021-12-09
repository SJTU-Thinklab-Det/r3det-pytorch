from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ml_nms_rotated_cuda',
    ext_modules=[
        CUDAExtension('ml_nms_rotated_cuda',
                      ['src/nms_rotated_cpu.cpp', 'src/nms_rotated_cuda.cu']),
    ],
    cmdclass={'build_ext': BuildExtension})
