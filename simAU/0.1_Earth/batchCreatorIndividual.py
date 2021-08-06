import sys
from datetime import datetime

def batchCreatorIndividual(path = "Batches"):
    now = datetime.now()
    months = ["jan", "feb", "march", "april", "may", "june", "july", "aug", "sept", "oct", "nov", "dec"]
    for i, j in enumerate(range(163)):
        i += 200
        i = str(i)
        i += "reds"
        message = f"#!/bin/bash\
\n\
\n#SBATCH -J SimAU_{months[now.month-1]}{now.day}_{j}\
\n#SBATCH -p general\
\n#SBATCH -o SimAU_{months[now.month-1]}{now.day}_{j}_%j.txt\
\n#SBATCH --mail-type=ALL\
\n#SBATCH --mail-user=taylor11@iu.edu\
\n#SBATCH --nodes=1\
\n#SBATCH --ntasks-per-node=1\
\n#SBATCH --cpus-per-task=3\
\n#SBATCH --time=36:00:00\
\n#SBATCH --mem=64G\
\n\
\npython3 massListSorter.py {i}\
\nsleep 5\
\npython3 simAUBatch.py --comboIndex {j} --date {months[now.month-1]}{now.day}\
\nsleep 5\
\nwait\
\npython3 massListSorter.py {i} Masslists/JupiterSimAU{months[now.month-1]}{now.day}Batch\
\n\
\ndate"
        if path: path += "/"
        with open(f"{path}zbatchSimAU{j}.script", "w") as file:
            file.write(message)

try:
	path = sys.argv[1]
except IndexError: 
	path = ""
batchCreatorIndividual(path)            
