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
        q = input("Make full list?")
        if q =='y': 
            return combo
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
