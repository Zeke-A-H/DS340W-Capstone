"""Run Python in the project Conda environment with native CAD DLL paths."""
from pathlib import Path
import os
import hashlib
import shutil
import subprocess
import sys
import tempfile


def build_environment(root, environment):
    env = os.environ.copy()
    cache = root / 'data' / '.cache'
    env.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
    env.setdefault('TORCH_EXTENSIONS_DIR', str(cache / 'torch_extensions'))
    env.setdefault('MAX_JOBS', '2')
    if (environment / 'bin' / 'nvcc.exe').is_file():
        cuda_path = environment
        if ' ' in str(environment):
            # PyTorch 2.5 writes an unquoted nvcc executable into Ninja files.
            # A junction avoids copying CUDA or moving the user's project.
            suffix = hashlib.sha256(str(environment).encode()).hexdigest()[:12]
            cuda_path = Path(tempfile.gettempdir()) / f'img2cad-cuda-{suffix}'
            if ' ' in str(cuda_path):
                raise RuntimeError('CUDA compilation needs a TEMP path without spaces')
            if not cuda_path.exists():
                command = f'cmd.exe /d /c mklink /J "{cuda_path}" "{environment}"'
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f'Could not create the CUDA path junction: {result.stderr}')
            if cuda_path.resolve() != environment.resolve():
                raise RuntimeError('The CUDA path junction points to a different environment')
        env['CUDA_HOME'] = str(cuda_path)
        env['CUDA_PATH'] = str(cuda_path)
        env['PATH'] = str(cuda_path / 'bin') + os.pathsep + env.get('PATH', '')
        # CUDA 11.8 supports MSVC 19.3x. Do not select newer 19.4x tools.
        for program_files in ('ProgramFiles', 'ProgramFiles(x86)'):
            base = Path(env.get(program_files, 'C:/Program Files'))
            for vcvars in base.glob('Microsoft Visual Studio/2022/*/VC/Auxiliary/Build/vcvarsall.bat'):
                vc_root = vcvars.parents[2]
                if not list((vc_root / 'Tools' / 'MSVC').glob('14.38.*')):
                    continue
                command = f'cmd.exe /d /s /c ""{vcvars}" x64 -vcvars_ver=14.38 >nul && set"'
                result = subprocess.run(command, capture_output=True, text=True, env=env)
                if result.returncode:
                    raise RuntimeError('Could not initialize the Visual C++ 14.38 build environment')
                for line in result.stdout.splitlines():
                    key, sep, value = line.partition('=')
                    if sep and key:
                        env[key.upper()] = value
                env['DISTUTILS_USE_SDK'] = '1'
                env['MSSdk'] = '1'
                return env
    return env


def main():
    root = Path(__file__).resolve().parents[1]
    environment = root / 'img2cad_conda'
    conda = shutil.which('conda.exe')
    if conda is None:
        candidate = Path.home() / 'anaconda3' / 'Scripts' / 'conda.exe'
        if candidate.is_file():
            conda = str(candidate)
    if conda is None:
        raise SystemExit('Conda not found. Run from an Anaconda prompt.')
    if not (environment / 'python.exe').is_file():
        raise SystemExit('The img2cad_conda environment has not been installed.')
    result = subprocess.run(
        [conda, 'run', '--prefix', str(environment), '--no-capture-output',
         'python', *sys.argv[1:]], cwd=root, env=build_environment(root, environment),
    )
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
