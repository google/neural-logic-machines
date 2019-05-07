# Blocks World

## Command
Please run these commands from the root directory of the project.

``` shell
# Train
$ jac-run scripts/blocksworld/learn_policy.py --task final
# Test
$ jac-run scripts/blocksworld/learn_policy.py --task final --test-only --load CHECKPOINT
# add [--test-epoch-size T] to control the number of testing cases.
```
