import numpy as np, sys, os

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

default = ["0.1_Earth", "4_SuperEarth", "9.8_SuperEarth", "Earth", "Half_Earth", "Jupiter"]

folders = [i for i in sys.argv]
if len(folders) == 1: 
    folders = default
else: 
    folders = folders[1:]
date = input("date >>>")

for folder in folders:
    message = f"{folder}:"
    for i in batchErrorList(f"{folder}/Masslists/{folder}{date}Batch.txt"):
        message += f" {i}"
    message += "\n"

