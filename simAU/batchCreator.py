import sys
from datetime import datetime

def batchCreator(path = ""):
    now = datetime.now()
    months = ["jan", "feb", "march", "april", "may", "june", "july", "aug", "sept", "oct", "nov", "dec"]
    for i, name in enumerate(["0-22", "23-45", "46-68", "69-91", "92-114", "115-137", "138-160", "161-162"]):
        dashIndex = name.index('-')
        nameMinus = name[:dashIndex]+ ".." + str((int(name[dashIndex+1:])-1))
        i += 100
        i = str(i)
        i += "reds"
        message = f"#!/bin/bash\
\n\
\n#SBATCH -J SimAU_{months[now.month-1]}{now.day}_{name}\
\n#SBATCH -p general\
\n#SBATCH -o SimAU_{months[now.month-1]}{now.day}_{name}_%j.txt\
\n#SBATCH --mail-type=ALL\
\n#SBATCH --mail-user=taylor11@iu.edu\
\n#SBATCH --nodes=1\
\n#SBATCH --ntasks-per-node=1\
\n#SBATCH --cpus-per-task=24\
\n#SBATCH --time=96:00:00\
\n#SBATCH --mem=128G\
\n\
\npython3 massListSorter.py {i}\
\nsleep 5\
\nfor i in {{{nameMinus}}}\
\ndo\
\n    python3 simAUBatch.py $i &\
\n    sleep 2\
\ndone\
\nsleep 5\
\npython3 simAUBatch.py {name[dashIndex+1:-1]+name[-1]}\
\nsleep 5\
\nwait\
\npython3 massListSorter.py {i} Masslists/JupiterSimAU{months[now.month-1]}{now.day}Batch\
\n\
\ndate"
        if path: path += "/"
        with open(f"{path}batchSimAU{name}.script", "w") as file:
            file.write(message)

try:
	path = sys.argv[1]
except IndexError: 
	path = ""
batchCreator(path)            
