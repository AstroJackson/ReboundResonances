import sys

sysArgs = dict()
for i, arg in enumerate(sys.argv):
    sysArgs[i] = arg

def copier(fileName, folderList =[]):
    if folderList == None:
        folderList = ["0.1_Earth", "4_SuperEarth", "9.8_SuperEarth", "Earth", "Half_Earth", "Jupiter"]
    with open(fileName) as file:
        contents = file.read()
    for folder in folderList:
        with open(f"{folder}/{fileName}", "w") as file:
            file.write(contents)

copier(fileName = sysArgs[1], folderList = sysArgs.get(2))
