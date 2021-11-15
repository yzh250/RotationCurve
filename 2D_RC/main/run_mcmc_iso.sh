#!/bin/bash
#SBATCH -p standard
#SBATCH --mem=10gb
#SBATCH --time=60:00:00
#SBATCH -o /scratch/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/7443_12705_mcmc_iso_log.log
#SBATCH -e /scratch/yzh250/Documents/UR_Stuff/Research_UR/RotationCurve/2D_RC/main/7443_12705_mcmc_iso_err.err
 
module load anaconda3/2020.11

srun python3 vel_map_MCMC_iso.py
