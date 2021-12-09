import os
import os.path as osp
import shutil
import sys
import warnings
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'r3det/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def make_cuda_ext(name, module, sources, sources_cuda=[]):
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def add_mim_extension():
    """Add extra files that are required to support MIM into the package.

    These files will be added by creating a symlink to the originals if the
    package is installed in `editable` mode (e.g. pip install -e .), or by
    copying from the originals otherwise.
    """

    # parse installment mode
    if 'develop' in sys.argv:
        # installed by `pip install -e .`
        mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        # installed by `pip install .`
        # or create source distribution by `python setup.py sdist`
        mode = 'copy'
    else:
        return

    filenames = ['tools', 'configs', 'demo', 'model-index.yml']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'r3det', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)

            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)

            if mode == 'symlink':
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                os.symlink(src_relpath, tar_path)
            elif mode == 'copy':
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    warnings.warn(f'Cannot copy file {src_path}.')
            else:
                raise ValueError(f'Invalid mode {mode}')


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='r3det',
        version=get_version(),
        description='Rotated Bounding Box Detection Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='zytx121',
        author_email='sjtu_zy@sjtu.edu.cn',
        keywords='computer vision, object detection',
        url='https://github.com/zytx121/r3det',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        ext_modules=[
            make_cuda_ext(
                name='nms_rotated_ext',
                module='r3det.ops.nms_rotated',
                sources=['src/nms_rotated_cpu.cpp', 'src/nms_rotated_ext.cpp'],
                sources_cuda=[
                    'src/nms_rotated_cuda.cu',
                    'src/poly_nms_cuda.cu',
                ]),
            make_cuda_ext(
                name='ml_nms_rotated_cuda',
                module='r3det.ops.ml_nms_rotated',
                sources=['src/nms_rotated_cpu.cpp'],
                sources_cuda=['src/nms_rotated_cuda.cu']),
            make_cuda_ext(
                name='box_iou_rotated_ext',
                module='r3det.ops.box_iou_rotated',
                sources=[
                    'src/box_iou_rotated_cpu.cpp',
                    'src/box_iou_rotated_ext.cpp'
                ],
                sources_cuda=['src/box_iou_rotated_cuda.cu']),
            make_cuda_ext(
                name='convex_ext',
                module='r3det.ops.convex',
                sources=['src/convex_cpu.cpp', 'src/convex_ext.cpp'],
                sources_cuda=['src/convex_cuda.cu']),
            make_cuda_ext(
                name='feature_refine_cuda',
                module='r3det.ops.fr',
                sources=['src/feature_refine_cuda.cpp'],
                sources_cuda=['src/feature_refine_kernel.cu']),
            make_cuda_ext(
                name='polygon_geo_cpu',
                module='r3det.ops.polygon_geo',
                sources=['src/polygon_geo_cpu.cpp']),
            make_cuda_ext(
                name='rbbox_geo_cuda',
                module='r3det.ops.rbbox_geo',
                sources=['src/rbbox_geo_cuda.cpp'],
                sources_cuda=['src/rbbox_geo_kernel.cu']),
            make_cuda_ext(
                name='rnms_ext',
                module='r3det.ops.rnms',
                sources=[
                    'src/rnms_ext.cpp', 'src/rcpu/rnms_cpu.cpp',
                    'src/rcuda/rnms_cuda.cpp'
                ],
                sources_cuda=['src/rcuda/rnms_kernel.cu']),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
