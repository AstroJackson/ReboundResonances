import numpy as np, sys, smtplib, os
from datetime import datetime

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
try:
    sys.argv[2]
    First = False
except IndexError:
    First = True

if not First:
    massListSorter(sys.argv[2])
    status = 'Ended'
    now = datetime.now()
    timerNumber = sys.argv[1] # if sys.argv[1] is the masslist file path, then sys.argv[2] is the timer number
    with open(f"timer{timerNumber}.txt") as file:
        month, day, hour, minute, second = [float(i) for i in file.read().split()]
    if month == now.month:
        timeHours = (now.day - day) * 24 + now.hour - hour + (now.minute - minute)/60 + (now.second- second)/3600
        hours, tempHours = int(timeHours), timeHours - int(timeHours)
        minutes, tempMinutes = int((tempHours) * 60), tempHours * 60 - int((tempHours) * 60)
        seconds = int(round((tempMinutes)*60, 0))
        extra = "Time taken: {:0>2}:{:0>2}:{:0>2}".format(hours, minutes, seconds)
    else:
        extra = "Time taken: different month"
    os.remove(f"timer{timerNumber}.txt")

else:
    status = 'Started'
    now = datetime.now()
    timerNumber = sys.argv[1]
    with open(f"timer{timerNumber}.txt", 'w') as file:
        file.write(f"{now.month} {now.day} {now.hour} {now.minute} {now.second}")
    extra = ""

gmail_user = 'jtpythontestemail@gmail.com'
gmail_password = 'PythonPassword1234!'

sent_from = gmail_user
to = ['jacksonisboss1@gmail.com']
subject = f'Simulation {status}!'
body = f"{status} at {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}\n{extra}"

email_text = """\
From: %s
To: %s
Subject: %s

%s
""" % (sent_from, ", ".join(to), subject, body)

try:
    smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtp_server.ehlo()
    smtp_server.login(gmail_user, gmail_password)
    smtp_server.sendmail(sent_from, to, email_text)
    smtp_server.close()
    print ("Email sent successfully!")
except Exception as ex:
    print ("Something went wrong….",ex)