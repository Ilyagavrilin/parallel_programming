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
    pkgs.python3
    pkgs.python3Packages.matplotlib
    pkgs.python3Packages.numpy
    pkgs.gdb
    pkgs.git
    pkgs.vim
  ];

  shellHook = ''
    export CC=${llvmPackages.clang}/bin/clang
    export CXX=${llvmPackages.clang}/bin/clang++
    echo "Welcome to development enviroment with Clang${clangVersion} and OpenMP"
  '';
}
