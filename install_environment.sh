#!/usr/bin/env bash
# =============================================================================
# One-command environment installation for ML4LB.
#
#   bash install_environment.sh [env-name] [--cpu|--gpu]
#   (default env name: mpc-test-01; default PyTorch build: auto-detected)
#
# Installs, into a conda environment:
#   - Python 3.8.17, numpy 1.21.2, SCIP 7.0.3, PySCIPOpt 3.1.1 (conda-forge)
#   - PyTorch 1.7.1 (CPU build, or CUDA 11.0 build on Linux GPU machines),
#     torch-scatter 2.0.7, torch-sparse 0.6.9, PyTorch Geometric 2.0.2
#   - GeCO 1.0.7, scipy 1.10.1, matplotlib 3.4.3, pandas 2.0.3,
#     pickleshare 0.7.5, pathlib 1.0.1
#   - the customized ecole 0.6.0, built from the vendored ./ecole folder
#
# Supported platforms: macOS (Intel and Apple Silicon via Rosetta 2) and
# Linux x86_64. PyTorch 1.7.1 has no Apple Silicon build, so on arm64 Macs
# the environment is created as x86_64 and runs transparently under Rosetta.
#
# GPU support (Linux only): if an NVIDIA GPU is detected (nvidia-smi), the
# CUDA 11.0 build of PyTorch 1.7.1 is installed instead of the CPU build;
# the bundled CUDA runtime works on any newer driver (e.g. a V100 with a
# CUDA 13.0 driver). Use --cpu or --gpu to override the auto-detection:
#
#   bash install_environment.sh [env-name] [--cpu|--gpu]
#
# The script is safe to re-run; it aborts if the environment already exists.
# =============================================================================
set -euo pipefail

# --- argument parsing: optional env name and --cpu/--gpu override -----------
ENV_NAME="mpc-test-01"
GPU_MODE="auto"   # auto | cpu | gpu
for arg in "$@"; do
    case "$arg" in
        --cpu) GPU_MODE="cpu" ;;
        --gpu) GPU_MODE="gpu" ;;
        -*)    printf 'ERROR: unknown option: %s\n' "$arg" >&2; exit 1 ;;
        *)     ENV_NAME="$arg" ;;
    esac
done
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECOLE_DIR="$REPO_DIR/ecole"
ECOLE_UPSTREAM_URL="https://github.com/ds4dm/ecole/archive/refs/tags/v0.6.0.tar.gz"
MINICONDA_DIR="$HOME/miniconda3"

# Use the official PyPI index (overriding any custom index configured in
# ~/.pip/pip.conf) and never let packages leak in from the user site-packages
# (~/.local), neither at build nor at run time of this script.
export PIP_INDEX_URL="https://pypi.org/simple"
export PYTHONNOUSERSITE=1

log()  { printf '\n=== %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "Step 1/7: checking prerequisites and detecting the platform"
# ---------------------------------------------------------------------------
command -v curl >/dev/null 2>&1 || fail "curl is required but not found."
command -v tar  >/dev/null 2>&1 || fail "tar is required but not found."
[[ -d "$ECOLE_DIR" && -d "$REPO_DIR/src" ]] \
    || fail "Run this script from a complete clone of the ML4LB repository (ecole/ and src/ not found)."

OS="$(uname -s)"
ARCH="$(uname -m)"
USE_X86_PREFIX=0    # 1: run build commands under 'arch -x86_64' (Apple Silicon)
CONDA_PLATFORM=""   # conda subdir override ('osx-64' on Apple Silicon)

# Runs a command, forcing x86_64 execution on Apple Silicon so that build
# tools (compilers, cmake, pip source builds) produce x86_64 binaries.
run_native() {
    if [[ "$USE_X86_PREFIX" == "1" ]]; then
        /usr/bin/arch -x86_64 "$@"
    else
        "$@"
    fi
}

# Runs conda with the platform override when one is needed.
run_conda() {
    if [[ -n "$CONDA_PLATFORM" ]]; then
        CONDA_SUBDIR="$CONDA_PLATFORM" "$CONDA_BIN" "$@"
    else
        "$CONDA_BIN" "$@"
    fi
}

case "$OS" in
    Darwin)
        xcode-select -p >/dev/null 2>&1 \
            || fail "Xcode Command Line Tools required: run 'xcode-select --install' first."
        if [[ "$ARCH" == "arm64" ]]; then
            # PyTorch 1.7.1 has no arm64 build: use an x86_64 (Rosetta) env.
            /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null \
                || fail "Rosetta 2 required: run 'softwareupdate --install-rosetta --agree-to-license' first."
            CONDA_PLATFORM="osx-64"
            USE_X86_PREFIX=1
            echo "Apple Silicon detected: creating an x86_64 environment (Rosetta 2)."
        fi
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-$ARCH.sh"
        ;;
    Linux)
        [[ "$ARCH" == "x86_64" ]] || fail "Only x86_64 Linux is supported (PyTorch 1.7.1 constraint)."
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        ;;
    *)
        fail "Unsupported operating system: $OS"
        ;;
esac

# Resolve the GPU mode. CUDA builds of PyTorch 1.7.1 exist for Linux only.
if [[ "$OS" == "Darwin" ]]; then
    [[ "$GPU_MODE" == "gpu" ]] && fail "--gpu is not available on macOS (PyTorch 1.7.1 has no macOS CUDA build); omit the flag or use --cpu."
    GPU_MODE="cpu"
elif [[ "$GPU_MODE" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        GPU_MODE="gpu"
        echo "NVIDIA GPU detected: installing the CUDA 11.0 build of PyTorch (override with --cpu)."
    else
        GPU_MODE="cpu"
    fi
fi
echo "platform: $OS $ARCH | pytorch mode: $GPU_MODE"

# ---------------------------------------------------------------------------
log "Step 2/7: locating (or installing) conda"
# ---------------------------------------------------------------------------
CONDA_BIN=""
for candidate in "$(command -v conda || true)" "$MINICONDA_DIR/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/miniconda3/bin/conda"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
done
if [[ -z "$CONDA_BIN" ]]; then
    echo "conda not found: installing Miniconda into $MINICONDA_DIR"
    curl -fsL "$MINICONDA_URL" -o /tmp/miniconda-installer.sh \
        || fail "failed to download the Miniconda installer from $MINICONDA_URL"
    bash /tmp/miniconda-installer.sh -b -p "$MINICONDA_DIR"
    rm -f /tmp/miniconda-installer.sh
    CONDA_BIN="$MINICONDA_DIR/bin/conda"
fi
echo "using conda: $CONDA_BIN ($("$CONDA_BIN" --version))"

# Abort if the environment already exists (check by name in 'conda env list').
if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fqx "$ENV_NAME"; then
    fail "conda env '$ENV_NAME' already exists. Remove it first: conda env remove -n $ENV_NAME"
fi

# ---------------------------------------------------------------------------
log "Step 3/7: creating conda env '$ENV_NAME' (python, numpy, SCIP, PySCIPOpt, cmake)"
# ---------------------------------------------------------------------------
# cmake < 4 is required: the vendored ecole 0.6.0 sub-projects use CMake
# minimums that CMake 4.x no longer accepts.
run_conda create -y -n "$ENV_NAME" -c conda-forge --override-channels \
    python=3.8.17 numpy=1.21.2 scip=7.0.3 pyscipopt=3.1.1 'cmake>=3.15,<4' make

# Resolve the environment prefix robustly (users may have custom envs_dirs).
ENV_PREFIX="$("$CONDA_BIN" env list | awk -v n="$ENV_NAME" '$1==n {print $NF; exit}')"
[[ -n "$ENV_PREFIX" && -x "$ENV_PREFIX/bin/python" ]] \
    || fail "could not locate the created environment '$ENV_NAME'."
PY="$ENV_PREFIX/bin/python"
echo "environment prefix: $ENV_PREFIX"

if [[ -n "$CONDA_PLATFORM" ]]; then
    # make future 'conda install' calls into this env resolve x86_64 packages
    "$CONDA_BIN" env config vars set "CONDA_SUBDIR=$CONDA_PLATFORM" -n "$ENV_NAME" >/dev/null
fi

# ---------------------------------------------------------------------------
log "Step 4/7: repairing the vendored ecole sources (if needed)"
# ---------------------------------------------------------------------------
# Two historical repo problems can leave the vendored ecole incomplete:
#   (a) '*.txt' files were tracked with git-lfs, but the LFS objects for the
#       six CMakeLists.txt files are missing from the remote, so fresh clones
#       only contain LFS pointer files;
#   (b) old .gitignore patterns 'data/' and 'instance/' excluded upstream
#       ecole directories from ever being committed.
# The affected files are UNMODIFIED upstream ecole v0.6.0 files (the sha256
# recorded in each LFS pointer matches upstream exactly), so they are safely
# restored from the upstream release tarball. The authors' customizations
# (observation functions etc.) live in other files and are never touched.

# Returns success if the file is a git-lfs pointer or does not exist.
needs_restore() { [[ ! -f "$1" ]] || grep -qs 'git-lfs.github.com' "$1"; }

needs_repair=0
for f in CMakeLists.txt libecole/CMakeLists.txt libecole/tests/CMakeLists.txt \
         libecole/benchmarks/CMakeLists.txt python/CMakeLists.txt docs/CMakeLists.txt; do
    needs_restore "$ECOLE_DIR/$f" && needs_repair=1
done
[[ -d "$ECOLE_DIR/libecole/src/instance" && -d "$ECOLE_DIR/libecole/include/ecole/data" ]] || needs_repair=1

if [[ "$needs_repair" == "1" ]]; then
    echo "vendored ecole is incomplete: restoring missing upstream v0.6.0 files"
    UP_TMP="$(mktemp -d)"
    curl -fsL "$ECOLE_UPSTREAM_URL" -o "$UP_TMP/ecole.tar.gz" \
        || fail "failed to download the upstream ecole 0.6.0 sources from $ECOLE_UPSTREAM_URL"
    tar -xzf "$UP_TMP/ecole.tar.gz" -C "$UP_TMP"
    UP="$UP_TMP/ecole-0.6.0"

    # (a) build files: when an LFS pointer is present, verify its sha256
    #     against the upstream file before restoring, so that customized
    #     files can never be silently overwritten.
    for f in CMakeLists.txt libecole/CMakeLists.txt libecole/tests/CMakeLists.txt \
             libecole/benchmarks/CMakeLists.txt python/CMakeLists.txt docs/CMakeLists.txt; do
        if needs_restore "$ECOLE_DIR/$f"; then
            if [[ -f "$ECOLE_DIR/$f" ]]; then
                want="$(grep '^oid sha256:' "$ECOLE_DIR/$f" | cut -d: -f2)"
                have="$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$UP/$f")"
                [[ "$want" == "$have" ]] \
                    || fail "sha256 mismatch for ecole/$f: the file was customized, cannot restore it from upstream."
            fi
            cp "$UP/$f" "$ECOLE_DIR/$f"
            echo "  restored ecole/$f"
        fi
    done

    # (b) directories lost to .gitignore: copy only if absent.
    for d in libecole/include/ecole/data libecole/include/ecole/instance \
             libecole/src/instance libecole/tests/data libecole/tests/src/data \
             libecole/tests/src/instance; do
        if [[ ! -d "$ECOLE_DIR/$d" ]]; then
            cp -R "$UP/$d" "$ECOLE_DIR/$d"
            echo "  restored ecole/$d/"
        fi
    done
    for f in LICENSE AUTHORS README.rst; do
        [[ -f "$ECOLE_DIR/$f" ]] || cp "$UP/$f" "$ECOLE_DIR/$f"
    done
    rm -rf "$UP_TMP"
else
    echo "vendored ecole sources are complete"
fi

# ---------------------------------------------------------------------------
log "Step 5/7: installing the PyTorch stack and Python libraries (pip)"
# ---------------------------------------------------------------------------
# pytest-runner first: the old torch-sparse setup.py needs it at build time.
run_native "$PY" -m pip install pytest-runner

# Select the PyTorch build. torch-scatter/torch-sparse contain CUDA kernels,
# so their wheel index must match the CUDA version of the torch build.
if [[ "$OS" == "Linux" && "$GPU_MODE" == "gpu" ]]; then
    # CUDA 11.0 build: supports the V100 (compute capability 7.0) natively and
    # runs on any newer NVIDIA driver via CUDA backward compatibility.
    TORCH_SPEC="torch==1.7.1+cu110"
    PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-1.7.1+cu110.html"
elif [[ "$OS" == "Linux" ]]; then
    TORCH_SPEC="torch==1.7.1+cpu"
    PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-1.7.1+cpu.html"
else
    TORCH_SPEC="torch==1.7.1"   # the macOS wheel is CPU-only
    PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-1.7.1+cpu.html"
fi

run_native "$PY" -m pip install "$TORCH_SPEC" -f https://download.pytorch.org/whl/torch_stable.html

# torch-scatter/torch-sparse versions matching torch 1.7.1 (Linux gets
# prebuilt wheels from the PyG index; macOS builds them from source).
run_native "$PY" -m pip install torch-scatter==2.0.7 torch-sparse==0.6.9 \
    -f "$PYG_WHEEL_INDEX"

run_native "$PY" -m pip install torch-geometric==2.0.2 geco==1.0.7 scipy==1.10.1 \
    matplotlib==3.4.3 pandas==2.0.3 pickleshare==0.7.5 pathlib==1.0.1

# Some transitive dependencies may bump numpy; re-pin it to the paper version
# BEFORE building ecole, whose bindings compile against the numpy C API.
run_native "$PY" -m pip install numpy==1.21.2

# ---------------------------------------------------------------------------
log "Step 6/7: building the customized ecole 0.6.0 from source"
# ---------------------------------------------------------------------------
CMAKE_ARCH_FLAG=""
[[ -n "$CONDA_PLATFORM" ]] && CMAKE_ARCH_FLAG="-DCMAKE_OSX_ARCHITECTURES=x86_64"

rm -rf "$ECOLE_DIR/build"
(
    cd "$ECOLE_DIR"
    # Environment tools (cmake, make, python) take precedence; the original
    # PATH is kept at the end so compilers installed outside /usr/bin (e.g.
    # RHEL devtoolset) remain reachable.
    export PATH="$ENV_PREFIX/bin:/usr/bin:/bin:$PATH"
    export CMAKE_PREFIX_PATH="$ENV_PREFIX"
    # shellcheck disable=SC2086  # CMAKE_ARCH_FLAG is intentionally unquoted
    run_native "$ENV_PREFIX/bin/cmake" -B build/ \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$PY" \
        $CMAKE_ARCH_FLAG
    run_native "$ENV_PREFIX/bin/cmake" --build build/ --parallel
    run_native "$PY" -m pip install --no-deps build/python
)

# ---------------------------------------------------------------------------
log "Step 7/7: verifying the installation"
# ---------------------------------------------------------------------------
export ML4LB_GPU_MODE="$GPU_MODE"
run_native "$PY" - <<'VERIFY'
import os
import warnings; warnings.filterwarnings("ignore")

import numpy, scipy, torch, torch_geometric, pyscipopt, matplotlib, pandas, geco, pickleshare, ecole
pins = {"numpy": (numpy, "1.21.2"), "scipy": (scipy, "1.10.1"), "torch": (torch, "1.7.1"),
        "torch_geometric": (torch_geometric, "2.0.2"), "pyscipopt": (pyscipopt, "3.1.1"),
        "matplotlib": (matplotlib, "3.4.3"), "pandas": (pandas, "2.0.3")}
for name, (mod, want) in pins.items():
    have = mod.__version__
    assert have.startswith(want.rsplit("+", 1)[0]) or have == want, f"{name}: expected {want}, got {have}"
    print(f"  {name:16s} {have}  OK")

# GPU availability (only asserted when the CUDA build was installed)
if os.environ.get("ML4LB_GPU_MODE") == "gpu":
    assert torch.cuda.is_available(), "CUDA build installed but torch.cuda.is_available() is False"
    print(f"  CUDA available: {torch.cuda.get_device_name(0)}  OK")

# SCIP solves a MIP
m = pyscipopt.Model(); m.hideOutput()
x = m.addVar(vtype="B"); y = m.addVar(vtype="B")
m.addCons(x + y <= 1); m.setObjective(-(x + 2 * y)); m.optimize()
assert m.getStatus() == "optimal" and m.getObjVal() == -2.0
print("  SCIP test solve  OK")

# customized ecole: instance generator + MilpBipartite observation
gen = ecole.instance.SetCoverGenerator(n_rows=40, n_cols=25, density=0.2); gen.seed(0)
env = ecole.environment.Configuring(scip_params={"presolving/maxrounds": 0},
                                    observation_function=ecole.observation.MilpBipartite())
env.seed(0)
obs, _, _, _, _ = env.reset(next(gen))
assert obs.variable_features.shape[1] == 10, "customized MilpBipartite feature count mismatch"
print("  ecole 0.6.0 (customized) observation  OK")

print()
print("Environment installed and verified successfully.")
VERIFY

echo ""
echo "Done. Activate the environment with:"
echo "    conda activate $ENV_NAME"
echo "and run the experiments from the repository root, e.g.:"
echo "    python src/evaluation_regression_k_prime.py --t_total=60 --dataset_id=0"
