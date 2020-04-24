module swap PrgEnv-intel PrgEnv-cray
module unload craype-haswell
module load craype-mic-knl
module load upcxx
module load cmake
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=CC ..
cmake --build .