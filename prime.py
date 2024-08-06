cnt = 3
maxnum = int(input("maxnum="))

if maxnum <= 1:
    print("No Prime")
elif maxnum >= 2:
    print("2")

while cnt < maxnum and maxnum >= 3:
    is_prime = True
    subcnt = 3
    while subcnt * subcnt <= cnt:
        if cnt % subcnt == 0:
            is_prime = False
            break
        subcnt += 2 
    if is_prime:
        print(cnt)
    cnt += 2
