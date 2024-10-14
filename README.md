# Parallel prgoramming tasks

This repo implememnts task for parallel programming course in MIPT (7 semestr)

## Build and Execution

All tasks associated with CMake files, build requires several libraries, to make it easier, please, use Nix (`nix-shell`) - it will create reproducible enviroment containing all libraries used inside.
How to install Nix:

```bash
$> sh <(curl -L https://nixos.org/nix/install) --daemon
```

To run enviroment:

```bash
$> nix-shell
```
