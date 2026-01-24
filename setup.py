from setuptools import setup, Extension
import glob
import os

ext_modules = []

if os.environ.get("TORQ_CUDA", "1") == "1":
    ctorq_extension = Extension(
        name="torq.cuda._C",
        sources=glob.glob("csrc/**/*.c", recursive=True),
        include_dirs=[
            "./csrc",
            "/usr/include/cuda/",
            "/usr/include/python3.12/"
        ],
        library_dirs=["/usr/local/cuda/lib64"],
    )
    ext_modules.append(ctorq_extension)

setup(
    name="torq",
    packages=["torq", "torq.cuda", "torq.core", "torq.compiler", "torq.compiler._fulqrum"],
    package_data={
        "torq": ["*.pyi", "py.typed"],
        "torq.cuda": ["*.pyi", "py.typed"]
    },
    ext_modules=ext_modules
)