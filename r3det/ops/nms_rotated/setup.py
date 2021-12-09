from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nms_rotated_ext',
    ext_modules=[
        CUDAExtension('nms_rotated_ext', [
            'src/nms_rotated_cpu.cpp', 'src/nms_rotated_cuda.cu',
            'src/nms_rotated_ext.cpp', 'src/poly_nms_cuda.cu'
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
