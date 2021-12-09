from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='box_iou_rotated_ext',
    ext_modules=[
        CUDAExtension('box_iou_rotated_ext', [
            'src/box_iou_rotated_cpu.cpp', 'src/box_iou_rotated_cuda.cu',
            'src/box_iou_rotated_ext.cpp'
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
