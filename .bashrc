# .bashrc

# The same home directories are used on multiple machines.  Some machines
# require slightly different configurations.  Please add your own
# configuration to the "Local definitions" section for the relevant machine.
# However, if the configuration is known to work on all machines then it can
# be written outside of the case statement below.

# Global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Local definitions

# Unload current cuda module with newer version
# Unload current gcc with newer version
# module unload gcc/4.8.5
# module load gcc/8.4.0
# More modules
# module unload python/3.6.5
# module load python/3.6.5
# module load intel/19.0.0.117
# module load sox/14.4.2


if [ 'screen' == "${TERM}" ]; then
  export PROMPT_COMMAND='printf "\e]2;%s %s\a" "${USER}" "${PWD}" '
fi

# Aliases
# alias alias_name="command_to_run"

alias cd_thesis="cd /srv/scratch/z5165205/Thesis"
alias openbashrc="vim ~/.bashrc"
alias act_venv="cd /srv/scratch/z5165205/Thesis; source thesis_env/bin/activate"


alias git_clean="git remote prune origin && git repack && git prune-packed && git reflog expire --expire=1.month.ago && git gc --aggressive"

alias ccache_clean="ccache -c"

# Find space taken by my accounts
alias diskspace="df -h | awk 'NR==1 || /z5165205/ || /chacmod/'"

# Find space taken by directories including hidden directories
alias dirspace="du -sch .[!.]* * |sort -h"

alias reloadbashrc=". ~/.bashrc"

alias jobs="qstat -u z5165205"

alias listproc="ps -u z5165205"

alias arthur="echo Arthur"

alias gogogo="python train_test_main.py 2>&1 | tee -a 'katanatest.txt'"

alias jobs="qstat -u z5165205"

alias 1g3h="qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=3:00:00"
alias 1g1h="qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=1:00:00"
alias 1g2h="qsub -I -l select=1:ncpus=8:ngpus=1:mem=46gb,walltime=2:00:00"
alias 4g3h="qsub -I -l select=1:ncpus=32:ngpus=4:mem=184gb,walltime=3:00:00"
alias 4g12h="qsub -I -l select=1:ncpus=32:ngpus=4:mem=184gb,walltime=2:00:00"
alias 4g12hbatch="qsub myjob.pbs.sh"


alias job65="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 65.pbs.sh"
alias job66="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 66.pbs.sh"
alias job67="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 67.pbs.sh"
alias job68="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 68.pbs.sh"
alias job69="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 69.pbs.sh"
alias job70="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 70.pbs.sh"
alias job71="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 71.pbs.sh"
alias job72="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 72.pbs.sh"
alias job73="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 73.pbs.sh"
alias job74="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 74.pbs.sh"
alias job75="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 75.pbs.sh"
alias job76="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 76.pbs.sh"
alias job77="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 77.pbs.sh"
alias job78="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 78.pbs.sh"
alias job79="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 79.pbs.sh"
alias job80="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub 80.pbs.sh"

alias jobc0="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c0.pbs.sh"
alias jobc1="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c1.pbs.sh"
alias jobc2="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c2.pbs.sh"
alias jobc3="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c3.pbs.sh"
alias jobc4="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c4.pbs.sh"
alias jobc5="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c5.pbs.sh"
alias jobc6="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c6.pbs.sh"
alias jobc7="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c7.pbs.sh"
alias jobc8="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c8.pbs.sh"
alias jobc9="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c9.pbs.sh"
alias jobc10="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c10.pbs.sh"
alias jobc11="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c11.pbs.sh"
alias jobc12="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c12.pbs.sh"
alias jobc13="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c13.pbs.sh"
alias jobc14="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c14.pbs.sh"
alias jobc15="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c15.pbs.sh"
alias jobc16="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c16.pbs.sh"
alias jobc17="cd /srv/scratch/z5165205/Thesis/job_scripts/; qsub c17.pbs.sh"

umask 0077