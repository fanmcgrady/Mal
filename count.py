import re
sec = "successfully"
sep = "select training sample"
with open("./log.txt", "r") as f:
    lines = f.readlines()
    cnt = 0
    eps = 0
    actions = 0
    for line in lines:
        if sec in line:
            cnt += 1
            for a in re.findall(r"\d", line):
                actions += int(a)
        if sep in line:
            eps += 1
            if eps%100==0:
                avg_actions = actions/100
                print(cnt, avg_actions)
                cnt=0
                actions = 0