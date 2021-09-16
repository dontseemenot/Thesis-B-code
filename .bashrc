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
alias 4g3h="qsub -I -l select=1:ncpus=32:ngpus=4:mem=184gb,walltime=2:00:00"
alias 4g12h="qsub -I -l select=1:ncpus=32:ngpus=4:mem=184gb,walltime=12:00:00"
alias 4g12hbatch="qsub myjob.pbs.sh"
umask 0077

