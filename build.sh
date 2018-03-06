export PYTHONPATH=/lustre/atlas/proj-shared/csc160/rbpittma_tensor/hvd_install/lib/python3.5/site-packages
module load singularity/2.4.0_legacy
# aprun -n1 -N1 singularity exec ../hvd_gather.img python3 setup.py build
aprun -n1 -N1 singularity exec ../hvd_gather.img python3 setup.py install --prefix=../hvd_install
