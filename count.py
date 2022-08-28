sec = "successfully"
sep = "select training sample"
with open("./log.txt", "r") as f:
    lines = f.readlines()
    cnt = 0
    eps = 0
    for line in lines:
        if sec in line:
            cnt += 1
        if sep in line:
            eps += 1
            if eps%100==0:
                print(cnt)
                cnt=0