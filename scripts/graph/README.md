# Graph Tasks, Shortest Path and Sorting

A set of graph-related reasoning tasks ans sorting task.

## Graph tasks

### Family tree tasks
``` shell
# Train: add --train-number 20 --test-number-begin 20 --test-number-step 20 --test-number-end 100
$ jac-run scripts/graph/learn_graph_tasks.py --task has-father
$ jac-run scripts/graph/learn_graph_tasks.py --task has-sister
$ jac-run scripts/graph/learn_graph_tasks.py --task grandparents --epochs 100 --early-stop 1e-7
$ jac-run scripts/graph/learn_graph_tasks.py --task uncle --epochs 200 --early-stop 1e-8
$ jac-run scripts/graph/learn_graph_tasks.py --task maternal-great-uncle --epochs 20 --epoch-size 2500 --early-stop 1e-8
# Test
$ jac-run scripts/graph/learn_graph_tasks.py --task TASK --test-only --load $CHECKPOINT
```
We use `loss < thresh` as the criteria for qualifying models.

### General graph tasks
``` shell
# Train
# AdjacentToRed
$ jac-run scripts/graph/learn_graph_tasks.py --task adjacent --gen-graph-colors 4
# 4-Connectivity
$ jac-run scripts/graph/learn_graph_tasks.py --task connectivity
# 6-Connectivity
$ jac-run scripts/graph/learn_graph_tasks.py --task connectivity --connectivity-dist-limit 6 --early-stop 1e-6 \
 --nlm-depth 8 --nlm-residual True --gen-graph-pmin 0.1 --gen-graph-pmax 0.3 --gen-graph-method dnc
# 1-Outdegree
$ jac-run scripts/graph/learn_graph_tasks.py --task outdegree --outdegree-n 1
# 2-Outdegree
$ jac-run scripts/graph/learn_graph_tasks.py --task outdegree --outdegree-n 2 \
--nlm-depth 5 --nlm-breadth 4 --nlm-residual True

# Test
$ jac-run scripts/graph/learn_graph_tasks.py --task TASK --test-only --load $CHECKPOINT \
--nlm-depth DEPTH --nlm-breadth BREADTH
```

### MNIST Input
We modified the `AdjacentToRed` task and replace the indicator of colors with MNIST digits. The NLM is integrated with LeNet and optimized jointly.

``` shell
$ jac-run scripts/graph/learn_graph_tasks.py --task adjacent-mnist \
--nlm-depth 2 --nlm-breadth 2 --nlm-attributes 16 --nlm-residual True --gen-graph-colors 10
```

## Shortest path

Below provides a proper set of parameters for this task.
``` shell
# Train
$ jac-run scripts/graph/learn_policy.py --task path
# Test
$ jac-run scripts/graph/learn_policy.py --task path --test-only --load $CHECKPOINT
```
For all available arguments see `jac-run scripts/graph/learn_policy.py -h`.

## Sorting

Below provides a proper set of parameters for this task.
``` shell
# Train
$ jac-run scripts/graph/learn_policy.py --task sort --nlm-depth 3 --nlm-breadth 2 \
--curriculum-graduate 10 --entropy-beta 0.01 \
--mining-epoch-size 200 --mining-dataset-size 20 --mining-interval 2
# Test
$ jac-run scripts/graph/learn_policy.py --task sort --test-only \
--nlm-depth 3 --nlm-breadth 2 --load $CHECKPOINT \
```
