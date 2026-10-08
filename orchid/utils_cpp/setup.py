from setuptools import Extension, setup

from setuptools.command.build_ext import build_ext
from setuptools import find_packages
import os
from pathlib import Path
import shutil


def get_pybind_include():
    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError(
            "pybind11 is required to build utils_cpp. "
            "Install it with 'pip install pybind11'."
        ) from exc
    return pybind11.get_include()


class BuildExt(build_ext):
    c_opts = {
        "msvc": ["/O2", "/std:c++17"],
        "unix": ["-O3", "-std=c++17"],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = list(self.c_opts.get(ct, []))
        lopts = list(self.l_opts.get(ct, []))

        # Detect CUDA automatically; UTILS_CPP_NO_GPU=1 forces a CPU build.
        nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
        if nvcc is None and "NVCC" not in os.environ and os.environ.get("CUDA_HOME"):
            nvcc = shutil.which(str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"))
        gpu_enabled = (
            os.environ.get("UTILS_CPP_NO_GPU") != "1"
            and os.environ.get("UTILS_CPP_ENABLE_GPU") != "0"
            and (nvcc is not None or os.environ.get("UTILS_CPP_ENABLE_GPU") == "1")
        )
        mode_file = Path(self.build_temp) / "gpu_mode"
        mode = "cuda" if gpu_enabled else "cpu"
        if not mode_file.is_file() or mode_file.read_text() != mode:
            self.force = True
        if gpu_enabled:
            if ct != "unix":
                raise RuntimeError("The utils_cpp CUDA build requires a Unix compiler")
            if nvcc is None:
                raise RuntimeError("UTILS_CPP_ENABLE_GPU=1 requires nvcc on PATH, NVCC, or CUDA_HOME")
            cuda_root = Path(os.environ.get("CUDA_HOME", Path(nvcc).resolve().parent.parent))
            cuda_lib = cuda_root / "lib64"
            if not cuda_lib.is_dir():
                cuda_lib = cuda_root / "lib"
            self.compiler.src_extensions.append(".cu")

        if ct == "unix":
            opts.append("-mavx2")
            if os.environ.get("UTILS_CPP_NO_OPENMP") != "1":
                opts.append("-fopenmp")
                lopts.append("-fopenmp")
        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args or []) + opts
            ext.extra_link_args = list(ext.extra_link_args or []) + lopts
            if gpu_enabled:
                ext.sources.append("src/cluster_legals_gpu_kernel.cu")
                ext.define_macros.append(("ENABLE_GPU", "1"))
                ext.include_dirs.append(str(cuda_root / "include"))
                ext.library_dirs.append(str(cuda_lib))
                ext.runtime_library_dirs.append(str(cuda_lib))
                ext.libraries.append("cudart")

        compile_cpp = self.compiler._compile

        def compile_object(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                preprocessor_args = [
                    arg for arg in cc_args if arg.startswith(("-I", "-D", "-U"))
                ]
                self.spawn([
                    nvcc, "-c", src, "-o", obj, "-O3", "-std=c++17",
                    "-Xcompiler=-fPIC", *preprocessor_args,
                ])
            else:
                compile_cpp(obj, src, ext, cc_args, extra_postargs, pp_opts)

        if gpu_enabled:
            self.compiler._compile = compile_object
        try:
            build_ext.build_extensions(self)
            mode_file.parent.mkdir(parents=True, exist_ok=True)
            mode_file.write_text(mode)
        finally:
            self.compiler._compile = compile_cpp


ext_modules = [
    Extension(
        "utils_cpp._cluster_legals",
        ["src/cluster_legals_pybind.cpp"],
        depends=[
            "src/cluster_legals_kernels.h",
            "src/predicate_eval_kernels.h",
            "src/topk_kernels.h",
            "src/cluster_legals_gpu_pybind.h",
            "src/cluser_legals_gpu_kernel.h",
        ],
        include_dirs=[get_pybind_include()],
        language="c++",
    )
]

setup(
    name="utils_cpp",
    version="0.1.0",
    description="C++ kernels for cluster legals",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
