set -e

env_name=$1
make_nproc=$2

eval "$(conda shell.bash hook)"
conda activate $env_name

root_dir=$(realpath $(dirname $0)/..)
build_dir=$root_dir/build
conda_dir=$(realpath $(dirname $(which python))/../)

mkdir -p $build_dir
cd $build_dir

cmake -D CMAKE_INSTALL_PREFIX=$root_dir \
      -D CUDA_TOOLKIT_ROOT_DIR=$conda_dir \
      -D CMAKE_CUDA_ARCHITECTURES=native \
      -D PKG_MOLECULE=on \
      -D PKG_CORESHELL=on \
      -D PKG_DIPOLE=on \
      -D PKG_DRUDE=on \
      -D PKG_KSPACE=on \
      -D PKG_REAXFF=on \
      -D PKG_QEQ=on \
      -D PKG_EXTRA-MOLECULE=on \
      -D PKG_MC=on \
      -D PKG_CLASS2=yes \
      -D BUILD_SHARED_LIBS=yes \
      -D BUILD_MPI=yes \
      -D BUILD_OMP=yes \
      -D MKL_INCLUDE_DIR=$conda_dir/include \
      -D CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
      -D CMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
      $root_dir/cmake


if [[ a =~ ^[0-9]+$ ]]; then
    make -j$make_nproc
else
    make
fi
