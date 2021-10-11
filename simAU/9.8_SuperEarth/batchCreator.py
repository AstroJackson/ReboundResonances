import sys, os
from datetime import datetime

def batchCreator(path = ""):
    presentDirectory = os.getcwd().split("/")[-1]
    maxHours = 40
    if presentDirectory == "Jupiter": maxHours = 56
    now = datetime.now()
    months = ["jan", "feb", "march", "april", "may", "june", "july", "aug", "sept", "oct", "nov", "dec"]
    direc = os.getcwd().split("/")[-1]
    for i, name in enumerate(["0-22", "23-45", "46-68", "69-91", "92-114", "115-137", "138-160", "161-162"]):
        dashIndex = name.index('-')
        nameMinus = name[:dashIndex]+ ".." + str((int(name[dashIndex+1:])-1))
        i += 100
        i = str(i)
        i += "reds"
        message = f"#!/bin/bash\
\n\
\n#SBATCH -J SimAU_{direc}_{months[now.month-1]}{now.day}_{name}\
\n#SBATCH -p general\
\n#SBATCH -o SimAU_{direc}_{months[now.month-1]}{now.day}_{name}_%j.txt\
\n#SBATCH --mail-type=ALL\
\n#SBATCH --mail-user=taylor11@iu.edu\
\n#SBATCH --nodes=1\
\n#SBATCH --ntasks-per-node=1\
\n#SBATCH --cpus-per-task=24\
\n#SBATCH --time={maxHours}:00:00\
\n#SBATCH --mem=128G\
\n\
\npython3 massListSorter.py {i}\
\nsleep 5\
\nfor i in {{{nameMinus}}}\
\ndo\
\n    python3 simAUBatch.py --comboIndex $i --date {months[now.month-1]}{now.day} &\
\n    sleep 2\
\ndone\
\nsleep 5\
\npython3 simAUBatch.py --comboIndex {name[dashIndex+1:-1]+name[-1]} --date {months[now.month-1]}{now.day}\
\nsleep 5\
\nwait\
\npython3 massListSorter.py {i} Masslists/{presentDirectory}{months[now.month-1]}{now.day}Batch\
\n\
\ndate"
        if path: path += "/"
        with open(f"{path}ybatchSimAU{name}.script", "w") as file:
            file.write(message)

try:
	path = sys.argv[1]
except IndexError: 
	path = ""
batchCreator(path)            
