# shell.nix
{ pkgs ? import <nixpkgs> {} }:

let
  clangVersion = "19";

  llvmPackages = pkgs.llvmPackages_19;

in pkgs.mkShell {
  buildInputs = [
    llvmPackages.clang
    pkgs.cmake
    llvmPackages.openmp
    pkgs.opencl-headers
    pkgs.opencl-clhpp
    pkgs.ocl-icd
    pkgs.intel-compute-runtime
    pkgs.intel-gpu-tools
    pkgs.python3
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.numpy
    pkgs.gdb
    pkgs.git
    pkgs.vim
    pkgs.ninja 
  ];

  shellHook = ''
    export CC=${llvmPackages.clang}/bin/clang
    export CXX=${llvmPackages.clang}/bin/clang++
    export OPENCL_VENDOR_PATH="${pkgs.intel-compute-runtime.outPath}/etc/OpenCL/vendors"
    export CMAKE_GENERATOR="Ninja"
    echo "${pkgs.opencl-clhpp.outPath}"
    echo "Welcome to development enviroment with Clang${clangVersion} and OpenMP"
  '';
}
