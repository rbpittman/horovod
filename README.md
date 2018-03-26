# Horovod Groups
In this package, the simple installation for Horovod (no GPU/NCCL
support) has been modified to support custom MPI groups, as well as a
Gather operation. 

# Example

In order to create 2 groups, one containing ranks 0, 1, and 2, and
another group containing ranks 2, 3, and 4, initialize HVD with:
> hvd.init([[0,1,2], [2,3,4]])

A subsequent allgather for group 0 might look like:
> all_images = hvd.allgather(tensor, group=0, name="my_allgather")
