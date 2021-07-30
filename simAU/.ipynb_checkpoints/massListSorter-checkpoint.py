import numpy as np, sys

def massListSorter(path):
    allData = []
    with open(path+".txt") as file:
        for i, line in enumerate(file):
            if i <= 1:
                continue
            splitted = line.split()
            splitted[-1] = float(splitted[-1])
            allData.append(splitted)
    lastEntry = [data[3] for data in allData]
    placement = [combo.index(i) for i in lastEntry]
    fullySorted = [0 for i in range(len(combo))]
    #fullySorted = [allData[i] for i in placement]
    for i, value in enumerate(placement): 
        fullySorted[value] = allData[i]
    with open(path+"SORTED.txt", 'w') as file:
        file.write("simAU\n"+"Inner Planet Mass\tOuter Planet Mass\tPercent Difference\tDistance\n")
        for i in fullySorted: 
            if type(i) != list:
                continue
            for j in i: 
                write = str(j)
                if j in outerDistances:
                    index = outerDistances.index(j)
                    write += "* ({}:{})".format(Info[index][0],Info[index][1])
                file.write(write+'\t')
            file.write('\n')

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

massListSorter("Masslists/test12")