import numpy as np, sys, os

class CustomException(Exception):
    pass

Info=[]
outerDistances = []
for i in range(1,11):
    for j in range(i+1,i+11):
        outerDist = .1*(j/i)**(2/3)
        if outerDist in outerDistances:
            continue
        pre = [i,j,.1, outerDist]
        Info.append(pre)
        outerDistances.append(outerDist)
copy = outerDistances.copy()
combo = list(np.linspace(.1, .5, 100)) + copy
combo.sort()

def batchErrorList(path):
    not_in = []
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        return []
    with open(path) as readFile:
        contents = readFile.read().split()[13::4]
    contents = [float(dist) for dist in contents]
    for dist in combo:
        if dist not in contents:
            not_in.append(dist)
    return not_in

months = ["jan", "feb", "march", "april", "may", "june", "july", "aug", "sept", "oct", "nov", "dec"]
default = ["0.1_Earth", "4_SuperEarth", "9.8_SuperEarth", "Earth", "Half_Earth", "Jupiter"]

date = ""
folders = [i for i in sys.argv]
if len(folders) == 2:
    if sys.argv[1] in [f"{mo}{da}" for mo in months for da in range(1,32)]:
        date = sys.argv[1]
        folders = default
if len(folders) == 1 or folders == default: 
    folders = default
    if not date: date = input("date >>>")
    if date not in [f"{mo}{da}" for mo in months for da in range(1,32)]: raise CustomException("Date entered was not in the correct format.")
else: 
    folders = folders[1:]
    date = ""

message = ""
for folder in folders:
    message = "#!/bin/bash\n\n"
    if folders == default: errorList =  batchErrorList(f"{folder}/Masslists/{folder}{date}Batch.txt")
    else: errorList = batchErrorList(folder)
    print(f"{folder}: {errorList}")
    for dist in errorList:
        message += f"sbatch ./zbatch{date}_{dist}.script\nsleep .3\n"
        timerNumber = np.random.random() * np.random.random() * 100 * 100 # highly unlikely two different distances will yield the same time number
        if folders == default:
            batchScript = f"#!/bin/bash\
\n\
\n#SBATCH -J SimAU_{folder}_{date}_{dist}\
\n#SBATCH -p general\
\n#SBATCH -o SimAU_{folder}_{date}_{dist}_%j.txt\
\n#SBATCH --mail-type=ALL\
\n#SBATCH --mail-user=taylor11@iu.edu\
\n#SBATCH --nodes=1\
\n#SBATCH --ntasks-per-node=1\
\n#SBATCH --cpus-per-task=3\
\n#SBATCH --time=48:00:00\
\n#SBATCH --mem=64G\
\n\
\npython3 massListSorter.py {timerNumber}\
\nsleep 5\
\npython3 simAUBatch{date}.py --comboIndex {combo.index(float(dist))} --date {date}\
\nsleep 5\
\nwait\
\npython3 massListSorter.py {timerNumber} Masslists/{folder}{date}Batch\
\n\
\ndate"
        with open(f"{folder}/zbatch{date}_{dist}.script", "w") as file:
            file.write(batchScript)
    with open(f"{folder}/zzerrorScripts{date}.script", "w") as file:
<<<<<<< HEAD
        file.write(f"{message}\ndate")




=======
        file.write(f"{message}\ndate")
>>>>>>> ddb56ad3e8e7bbe77db8a6c4e7d41cbdabd880a8
