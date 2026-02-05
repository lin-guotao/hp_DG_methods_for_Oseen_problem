# hp_DG_methods_for_Oseen_problem
### Numerical experiments for the paper: "hp-version discontinuous Galerkin methods for the Oseen problem on essentially arbitrarily shaped elements."

**This repository is unmaintained and is provided solely for the purpose of reproducing the numerical results in the associated paper.**

---

## Prerequisites

### Option A: Using Docker (Recommended)
You need to have **Docker** installed on your system. 
* [Download Docker](https://www.docker.com/)

### Option B: Local Installation (Without Docker)
If you prefer not to use Docker, you must manually install **polydeal**:
* [polydeal Repository](https://github.com/fdrmrc/Polydeal)

**Important:** Please ensure this repository's folder is placed in the **same parent directory** as the `polydeal` folder for the build system to locate dependencies.

---

## Basis
This implementation is based on **[deal.II](https://www.dealii.org/)** and **[polydeal](https://github.com/fdrmrc/Polydeal)**.

---

## Usage

Follow these steps to build and run the code:

### 1. Clone the repository
```bash
git clone https://github.com/lin-guotao/hp_DG_methods_for_Oseen_problem.git oseen_code
cd oseen_code
```
### 2. Set up the Docker environment (Only if using Docker)
If you are using Docker, pull the required image and start the container. Skip this step if you have installed polydeal locally.
```bash
docker pull docker.io/polyoseen/dealii-polydeal:latest
docker run --privileged -ti -v $(pwd):/home/dealii/shared -w /home/dealii/shared docker.io/polyoseen/dealii-polydeal:latest
```
### 3. Build the project
Inside the container (or in your local project folder), run:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
### 4. Run the numerical experiments
After a successful build, the executable files for various numerical experiments will be located in the build/examples directory.
For example, to run the rectangular domain test:
```bash
cd examples
./example1_rect
```
Results: All output files are saved in build/examples/output.
