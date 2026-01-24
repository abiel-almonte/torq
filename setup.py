from setuptools import setup, Extension
import glob


ctorq_extension = Extension(
    name="torq.cuda._C",
    sources=glob.glob("csrc/**/*.c", recursive=True),
    include_dirs=[
        "./csrc",
        "/usr/include/cuda/",
        "/usr/include/python3.12/"
        #cuda_runtime.h is already in /usr/include/
    ],
    library_dirs=["/usr/local/cuda/lib64"],
)

setup(
    name="torq",
    packages=["torq", "torq.cuda", "torq.core", "torq.compiler", "torq.compiler._fulqrum"],
    package_data={
        "torq": ["*.pyi", "py.typed"],
        "torq.cuda": ["*.pyi", "py.typed"]
    },
    ext_modules=[ctorq_extension]
)