#!/bin/bash

#SBATCH -J SIMAU_July27TEST
#SBATCH -p general
#SBATCH -o SIMAU_July27TEST_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=taylor11@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --mem=1G


for i in {0..161} #162 times, not 163
do
	python3 simAU1.py $i
done
python3 simAU1.py 162 #163rd time does not include a '&' nor a 'sleep' command