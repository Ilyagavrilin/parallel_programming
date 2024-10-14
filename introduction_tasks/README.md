# Initial tasks

This folders contains inital OpenMP tasks, follow rules, how to build and run the code.

## Build and Run
>
> [!IMPORTANT]
> To succesfully build the task it is better to use `nix-shell`.  
> Take a look on main README.md

```bash
$> nix-shell
$> cd task_$TASK_NUMBER
$> cmake -B build
# Add flag -DENABLE_TIMING=ON to enable time measurement in runtime
$> ./build/bin/...
```
