import random
# 1から100までのランダムな数を生成
target_number = random.randint(100, 999)

first_num = int(target_number/100)
second_num = (int(target_number/10))%10
third_num = target_number % 10

# 正解するまで繰り返す
while True:
    guess = int(input("数を入力してください: "))

    same = 0
    diff = 0

    first_guess = int(guess/100)
    second_guess = (int(guess/10))%10
    third_guess = guess%10

    if first_guess == first_num:
        same += 1
    elif first_guess == second_num or first_guess == third_num:
        diff += 1

    if second_guess == second_num:
        same += 1
    elif second_guess == first_num or second_guess == third_num:
        diff += 1

    if third_guess == third_num:
        same += 1
    elif third_guess == first_num or third_guess == second_num:
        diff += 1

    if same == 3:
        print("正解です")
        break

    else:
        print("same=",same)
        print("different",diff)
        