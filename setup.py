import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    setup(
        name='UniTraj',
        version=1.0,
        description='A Unified Framework for Scalable Vehicle Trajectory Prediction',
        author='Lan Feng',
        author_email='fenglan18@outlook.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },

        ext_modules=[
            make_cuda_ext(
                name='knn_cuda',
                module='unitraj.models.mtr.ops.knn',
                sources=[
                    'src/knn.cpp',
                    'src/knn_gpu.cu',
                    'src/knn_api.cpp',
                ],
            ),
            make_cuda_ext(
                name='attention_cuda',
                module='unitraj.models.mtr.ops.attention',
                sources=[
                    'src/attention_api.cpp',
                    'src/attention_func_v2.cpp',
                    'src/attention_func.cpp',
                    'src/attention_value_computation_kernel_v2.cu',
                    'src/attention_value_computation_kernel.cu',
                    'src/attention_weight_computation_kernel_v2.cu',
                    'src/attention_weight_computation_kernel.cu',
                ],
            ),
        ],
    )
