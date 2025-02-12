## Prepare Conda Environment
```bash
conda env create -f environment.yaml -n <env_name>
```
where `<env_name>` is the name of your target conda environment.

## Compile Lammps with Pair XequiNet
```bash
bash script/build.sh <env_name> <nproc>
```
where `<nproc>` is the number of threads to run build process.

## Note
Your cmake version must be between 3.16 and **3.24**.

## Usage
```
pair_style   xequinet/xequinet32/xequinet64
pair_coeff   * * <model>.jit <element1> <element2> ... <elementn>
```
where `xequinet` and `xequinet32` is for FP32, and `xequinet64` is for FP64; `<model>.jit` is the path for the jit compiled model. `<elementi>`s are the element types in your LAMMPS input file, and the order must be consistent with your input file.

And Remember to export the library in your job scripts.
```bash
source activate archetype

libtorch_path=$(realpath $(dirname $(python -c 'import torch; print(torch.__file__)')))/lib
conda_lib_path=$(realpath $(dirname $(which python))/..)/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/.../path_to_lammps/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$libtorch_path:$conda_lib_path

```