1. Connect to Niagara: DUO and password

2. Go Mist:
ssh -Y mist-login01
cd $SCRATCH

3. Load Virtual Env
module load anaconda3
source activate fengsanz1


4. Submit Job:
sbatch jobscript.sh

5. Monitoring Jobs
squeue to show the job queue (squeue -u $USER for just your jobs)

squeue -j JOBID to get information on a specific job (alternatively, scontrol show job JOBID, which is more verbose),

squeue --start -j JOBID to get an estimate for when a job will run.

jobperf JOBID gives the instantaneous view of the cpu+memory usage of a running job,

scancel JOBID to cancel the job, or scancel -u USERID to cancel all your jobs (be careful!)

sacct to get information on your recent jobs,



--------------------------------------------------------------------------------------------------------------------
6. Set up virtual env:
Following are the steps to create a virtual env and remove/other useful things you may want to check.

*For Niagara:
    Create Virtual Env for yourself:

    module load NiaEnv/2019b python/3.11.5
    mkdir ~/.virtualenvs
    virtualenv --system-site-packages ~/.virtualenvs/myenv


    Activate:
    source ~/.virtualenvs/myenv/bin/activate 

    Deactivate: 
    deactivate


For Mist: 
    Create Virtual Env:
    module load anaconda3
    conda create -n fengsanz1 python=3.9

    Activate: 
    source activate fengsanz1

    To deactivate:
    source deactivate

    Remove:
    conda remove --name fengsanz1 --all


Install Pytorch
    source activate fengsanz1
    conda config --prepend channels /scinet/mist/ibm/open-ce-1.9.1
    conda config --set channel_priority strict
    conda install -c /scinet/mist/ibm/open-ce-1.9.1 pytorch=2.0.1 cudatoolkit=11.8


After Install, clean cache
    conda clean -y --all
    rm -rf $HOME/.conda/pkgs/*
    rm -f $HOME/.condarc


Now you should have Cuda, and Pytorch ready. Cuda version: 11.8. 
