# print(5)
# print('풍선')
# print("나비")
# print("ㅋ"*6)
# print(5 > 10)
# print(True)
# print(not False)
# print(not (5>14))

s1 = "사당"
print(s1 +"행 열차가 들어오고 있습니다.")

print(abs(-5))
print(pow(4,2)) #4^2
print(max(5,2))
print(min(5,2))
print(round(3.14))
print(round(4.99))

from math import *
print(floor(4.99)) #내림
print(ceil(3.14)) #올림
print(sqrt(16)) #제곱근


from random import *
print(random()) # 0.0 이상 1.0 미만의 임의의 값 생성
print(random() * 10)
print(int(random() * 10)) # 0부터 10미만의 임의의 값 생성

print(int(random() * 45) + 1) # 1부터 45이하의 임의의 값 생성
print(randrange(1,46)) # 1부터 46미만의 임의의 값 생성
print(randint(1,45)) # 1부터 45이하의 임의의 값 생성

jumin = "990120-1234567"
print("성별 : " + jumin[7])
print("연 : " + jumin[0:2]) #0부터 2 직전까지
print("생년월일 : " + jumin[:6]) #처음부터 6 직전까지
print("뒷자리 : " + jumin[7:]) #7부터 마지막까지
print("뒤 7자리 (뒤에서부터) : " + jumin[-7:]) # 맨뒤에서 7번째부터 끝까지

python = "Pytion is Amazing"
print(python.lower())
print(python.upper())
print(python[0].isupper()) #대문자인지 아닌지
print(len(python))
print(python.replace("Pytion", "Java")) #대체

index = python.index("n") #n 이라는 문자가 몇번째 위치에 있는지 (최초 하나)
print(index)
index = python.index("n", index + 1) #시작 위치가 index 위치 + 1, 두번째 n 찾기
print(index)
print(python.find("n"))  #원하는 값이 있는 값의 인덱스 반환(최초 하나)
print(python.find("Java"))  #원하는 값이 없을땐 -1
#print(python.index("Java"))  #원하는 값이 없을땐 오류
print(python.count("n")) #원하는 문자가 몇번 나오는지

print("a" + "b")
print("a", "b")
print("나는 %d살 입니다." %20) 
print("나는 %s을 좋아해요." %"파이썬")
print("Apple은 %c로 시작해요" %"A")
print("나는 %s색과 %s색을 좋아해요" %("파랑", "빨간"))

print("나는 {}살 입니다." .format(20))
print("나는 {}색과 {}색을 좋아해요" .format("파랑", "빨간"))
print("나는 {0}색과 {1}색을 좋아해요" .format("파랑", "빨간"))
print("나는 {1}색과 {0}색을 좋아해요" .format("파랑", "빨간"))
print("나는 {age}살이며, {color}색을 좋아해요" . format(age = 20, color="빨간"))

age = 20
color = "빨간"
print(f"나는 {age}살이며, {color}색을 좋아해요")

print("\n")
print('저는 "나도코딩" 입니다')
print("저는 \"나도코딩\" 입니다")

print("Red Apple\rPine") #커서를 맨 앞으로 이동 덮어쓰기
print("Redd\b apple") # 앞 글자 삭제

site = "http://naver.com"
print(site[7:10] + str(len(site[7:12])) + str(site[7:12].count("e")) + "!")
print("\n")
my_str = site.replace("http://", "")
print(my_str)
my_str = my_str[:my_str.index(".")]
print(my_str)
password = my_str[:3] + str(len(my_str)) + str(site[7:12].count("e")) + "!"
print("{0}의 비밀번호는 {1}입니다." .format(site, password))

#리스트
print("\n")

subway1 = 10
subway2 = 10
subway3 = 10
subway = [10 ,20 ,30]
print(subway)
subway = ["유재석" ,"조세호" ,"박명수"]
print(subway)
print(subway.index("조세호")) #리스트 인덱스 반환

subway.append("하하")
print(subway)
subway.insert(1, "정형돈") #중간에 삽입
print(subway)


print(subway.pop()) #pop 연산 뒤에서 하나씩 꺼냄 (원래값 사라짐)
print(subway.pop()) #pop 연산

print(subway)
print(subway.count("유재석")) #해당 원소 갯수 새기
print("\n")

num_list = [5,4,3,2,1]
num_list.sort() # 정렬
print(num_list)

num_list.reverse() # 거꾸로
print(num_list)

num_list.clear() # 리스트 비우기
print(num_list)

#다양한 자료형과 함께 사용
mix_list = ["조세호", 20 ,True]
print(mix_list)

num_list = [5,4,3,2,1]   #합치기 가능
num_list.extend(mix_list)
print(num_list)

#사전 자료형 dictionary

cabinet = {3:"유재석", 100:"김태호"}
print(cabinet[3]) # 키 값으로 사전을 찾음
print(cabinet[100]) #값이 없으면 오류 

print(cabinet.get(3))
print(cabinet.get(5)) #없으면 Nome 출력
print(cabinet.get(5, "사용 가능")) #없으면 "사용 가능" 출력

print(3 in cabinet) #True,  in 을 통해 키가 있는지 확인 가능 
print(5 in cabinet) #Flase

cabinet = {"A-3":"유재석", "B-100":"김태호"} # 키값은 문자열도 가능
print(cabinet["A-3"])
print(cabinet["B-100"])

print(cabinet)
cabinet["C-20"] = "조세호" # 값 추가
print(cabinet)
cabinet["C-20"] = "김종국" # 중복일 경우 업데이트
print(cabinet)

del cabinet["A-3"] # 삭제
print(cabinet)

print(cabinet.keys()) #키값만 출력
print(cabinet.values()) #value 출력

print(cabinet.items()) # key- value 쌍으로 출력

cabinet.clear() #딕셔너리 초기화
print(cabinet)

#튜플 -> 값을 추가 및 변경 불가능
menu = ("돈까스", "치즈까스")
print(menu[0])
print(menu[1])
#menu.add("생선까스") #값 추가 불가능

(name, age, hobby) = ("김종국", 22, "코딩")
print(name, age, hobby)

#집합 (set) -> 중복 안됨, 순서 없음
my_set = {1,2,3,3,3}
print(my_set)

java = {"유재석", "김태호", "양세형"}
python = set(["유재석", "박명수"])

print(java & python) #교집합
print(java.intersection(python))

print(java | python) #합집합
print(java.union(python))

print(java - python)
print(java.difference(python)) #차집합

python.add("김태호") #set에 값 추가
print(python)

java.remove("김태호") # 값 삭제
print(java)

#타입 바꾸기

menu = {"커피", "우유", "주스"}
print(menu, type(menu))

menu = list(menu)
print(menu, type(menu))

menu = tuple(menu)
print(menu, type(menu))

menu = set(menu)
print(menu, type(menu))

from random import *
lst = [1,2,3,4,5]
print(lst)
shuffle(lst) #lst를 무작위로 섞음
print(lst)
print(sample(lst, 1)) #lst에서 1개를 무작위로 뽑음.

dat = range(1,21) #1부터 21직전까지 숫자 생성 (rage 타입이므로 자료형 변경하기)
print(dat, type(dat))
dat = list(dat)
print(dat, type(dat))

shuffle(dat)
winners = sample(dat, 4)
print(" -- 당첨자 발표 --")
print(" -- 치킨 당첨자 : {0} --" .format(winners[0]))
print(" -- 커피 당첨자 : {0} --" .format(winners[1:]))
print(" -- 축하합니다 --")

print("\n")
#조건문
#whether = input("오늘 날씨는 어때요?") #입력값 문자열로 저장
whether = "비"
if whether == "비" or whether == "눈":
    print("우산을 챙기세요")
elif whether == "미세먼지":
    print("마스크를 챙기세요")
else:
    print("준비물 필요 없어요")

temp = 4
#temp = int(input("기온은 어때요?")) #int로 변환
if 30 <= temp:
    print("나가지 마세요")
elif 10 <= temp and temp < 30:
    print("괜찮은 날씨에요")
elif 0 <= temp < 10:
    print("외투를 챙기세요")
else:
    print("너무 추워요 나가지 마세요")

#for
for waiting_no in [1,2,3,4,5]:
    print("대기번호 : {0}" .format(waiting_no))

for waiting_no in range(1,6):
    print("대기번호 : {0}" .format(waiting_no))

starbucks = ["아이언맨", "토르", "아이엠 그루트"]
for customer in starbucks:
    print("{0}, 커피 준비되었습니다." .format(customer))

#while
customer = "토르"
index = 5
while index >=1: #while True: 는 무한반복문, ctrl + c 강제종료
    print("{0}, 커피 준비되었습니다. {1}번 남았어요 " .format(customer, index))
    index -= 1
    if index == 0:
        print("커피는 폐기처분 되었습니다")

# customer = "토르"
# person = "Unknown"
# while person != customer:
#     print("{0}, 커피 준비되었습니다." .format(customer))
#     person = input("이름이 어떻게 되세요")

absent = [2,5]
nobook = [7]
for student in range(1,11):
    if student in absent:
        continue #이 이하로 실행하지 않고 다음 반복으로 감: 스킵
    elif student in nobook:
        print("오늘은 여기까지. {0}은 교무실로" .format(student))
        break #반복문 종료
    print("{0}, 책을 읽어봐" .format(student))

#한줄 for 문
student = [1,2,3,4,5]
print(student)
student = [i+100 for i in student]
print(student)

student = ["Iron man", "Thor", "I am groot"]
student = [len(i) for i in student]
print(student)

student = ["Iron man", "Thor", "I am groot"]
student = [i.upper() for i in student]
print(student)

print("\n") 

from random import *
cnt = 0
for i in range(1,51):
    time = randrange(5,51)
    if 5 <= time and time <= 15:
        cnt += 1
        print("[0] {0}번째 손님 (소요시간 : {1}분)" .format(i, time))
    else:
        print("[ ] {0}번째 손님 (소요시간 : {1}분)" .format(i, time))

print("총 탑승 승객 : {0}분" .format(cnt))

#함수
print("\n") 

def open_account():
    print("새로운 계좌가 생성되었습니다")

def deposite(balance, money):
    print("입금이 완료되었습니다. 잔액은 {0}원입니다." .format(balance+money))
    return balance + money

def withdraw(balance, money):
    if balance >= money:
        print("출금이 완료되었습니다. 잔액은 {0}원입니다." .format(balance - money))
        return balance - money
    else:
        print("출금이 완료되지 않았습니다. 잔액은 {0}원 입니다." .format(balance))
        return balance

def withdraw_night(balance, money):
    commission = 100
    return commission, balance - money - commission

balance = 0
balance = deposite(balance, 1000)
print(balance)
balance = withdraw(balance, 2000)
print(balance)
commission, balance = withdraw_night(balance, 500)
print("수수료는 {0}원이며, 잔액은 {1}원 입니다." . format(commission, balance))

#기본값
def profile(name, age=17, main_lang="파이썬"):
    print("이름 :{0}\t 나이: {1}\t 언어: {2}"\
        .format(name, age, main_lang))

profile("유재석")
profile("김태호")

#키워드값
def profile(name, age, main_lang):
    print(name, age, main_lang)

profile(name="유재석", main_lang="파이썬", age=20)
profile(main_lang="자바", age=24, name="김태호")  #순서는 상관 없다

print("\n")
#가변 인자
def profile1(name, age, lang1, lang2, lang3, lang4, lang5):
    print("이름 : {0}\t나이 : {1}\t" .format(name, age), end=" ")
    print(lang1, lang2, lang3, lang4, lang5)

profile1("유재석", 20 ,"Python", "Java", "C", "C++", "C#")
profile1("김태호", 25, "Kotlin", "Swift", "", "", "")

def profile1(name, age, *language):
    print("이름 : {0}\t나이 : {1}\t" .format(name, age), end=" ")
    for lang in language:
        print(lang, end=" ")
    print()

profile1("유재석", 20 ,"Python", "Java", "C", "C++", "C#", "JavaSCript")
profile1("김태호", 25, "Kotlin", "Swift")

print()
#지역변수 전역변수
gun = 10
def checkpoint(soldiers):
    gun = 20 
    gun = gun - soldiers
    print("[함수내] 남은 총 : {0}" .format(gun))
print("전체 총 : {0}" .format(gun))
checkpoint(2)
print("남은 총 : {0}" .format(gun))

gun = 10
def checkpoint(soldiers):
    global gun #전역 공간에 있는 gun 사용
    gun = gun - soldiers
    print("[함수내] 남은 총 : {0}" .format(gun))
print("전체 총 : {0}" .format(gun))
checkpoint(2)
print("남은 총 : {0}" .format(gun))

def checkpoint_return(gun, soldiers): 
    gun = gun - soldiers
    print("[함수내] 남은 총 : {0}" .format(gun))
    return gun #리턴해주므로 함수 내부 값 전달 가능
print("전체 총 : {0}" .format(gun))
checkpoint(2)
print("남은 총 : {0}" .format(gun))

def std_weight(height, gender):
    if gender =="남자":
        return round((height*height*22),2)
    else:
        return round((height*height*21),2)
# gender = input("성별을 입력하세요.")
# height = int(input("키를 입력하세요 : "))/100
gender = "남성"
height = 180/100
print("키 {0}cm {1}의 표준 체중은 {2}Kg 입니다." .format(height, gender, std_weight(height, gender)))

print("Python", "Java", sep=" ") #sep 을 통해 콤마 사이를 구분해주는 문자를 지정할 수 있음
print("Python", "Java", sep=" ", end = "?") #end를 통해 문장이 끝나는 문자를 지정할 수 있음
print("무엇이 더 재미있을까요")

import sys
print("Python", "Java", file = sys.stdout) #표준 출력 
print("Python", "Java", file = sys.stderr) #표준 에러 출력

scores = {"수학":0, "영어":50, "코딩":100}
for subject, score in scores.items():
    print(subject, score)

for subject, score in scores.items():
    print(subject.ljust(8), str(score).rjust(4), sep=":")

    scores = {"수학":0, "영어":50, "코딩":100}
for subject, score in scores.items():
    print(subject.ljust(8), str(score).rjust(4))

for num in range(1,21):
    print("대기번호 : " + str(num).zfill(3)) #3크기의 공간 확보 없는값은 0으로 채움

# answer = input("아무값이나 입력하세요 : ") #사용자입력을 통해 값을 받게되면 항상 문자열 형태로 값을 받게 된다.
# print(type(answer))
# print("입력하신 값은" + answer +"입니다.")
# print()

#빈 자리는 빈 공간으로 두고, 오른쪽 정렬을 하되, 총 10자리 공간을 확보
print("{0: >10}" .format(500))

#양수일땐 +로 표시, 음수일땐 -로 표시
print("{0: >+10}" .format(500))

#왼쪽 정렬하고, 빈칸을 _로 채움
print("{0:_<+10}" .format(500))

# 3자리마다 콤마를 찍어주기
print("{0:,}" .format(100000))

# 부호 추가
print("{0:+,}" .format(100000))

#세 자리마다 콤마를 찍어주고 왼쪽정렬, 부호도 붙이고, 자릿수 확보, 빈자리는^로 채움
print("{0:^<+30,}" .format(10000000))

#소수점 출력
print("{0:f}" .format(5/3))

#소수점 특정 자리수까지만 표시
print("{0:.2f}" .format(5/3))

#파일 입출력
#덮어쓰기 wirte
score_file = open("score.txt", "w", encoding="utf8")
print("수학 : 0", file=score_file)
print("영어 : 50", file=score_file)
score_file.close()

#이어쓰기 append
score_file = open("score.txt", "a", encoding="utf8")
score_file.write("과학 : 80")
score_file.write("\n코딩 : 100") #줄바꿈 해줘야함
score_file.close()

#파일 한번에 읽기 read
score_file = open("score.txt", "r", encoding="utf8")
print(score_file.read())
score_file.close

#한줄씩 찾아와 읽기
score_file = open("score.txt", "r", encoding="utf8")
print(score_file.readline(), end="") #줄별로 읽고 커서는 다음줄로 이동 해놓음
print(score_file.readline(), end="") #줄별로 읽고 커서는 다음줄로 이동 해놓음
print(score_file.readline(), end="") #줄별로 읽고 커서는 다음줄로 이동 해놓음
print(score_file.readline()) #줄별로 읽고 커서는 다음줄로 이동 해놓음
score_file.close

#몇 줄인지 모를때
score_file = open("score.txt", "r", encoding="utf8")
while True:
    line = score_file.readline()
    if not line:
        break
    print(line, end="")
score_file.close()

print()

#리스트에 넣어 처리하기
score_file = open("score.txt", "r", encoding="utf8")
lines = score_file.readlines() #list 형태로 저장
for line in lines: #리스트에서 하나씩 가져와 출력
    print(line, end="")
score_file.close

print()
#pickle
import pickle
profile_file = open("profile.pickle", "wb") #wirte binary
profile = {"이름":"박명수", "나이":30, "취미":["축구", "골프", "코딩"]}
#print(profile)
pickle.dump(profile, profile_file) #profile에 있는 정보를 profile_File에 저장
profile_file.close

profile_file = open("profile.pickle", "rb") #read binary
profile = pickle.load(profile_file) #profile_file정보 profile에 불러오기
print(profile)
profile_file.close()

print()
import pickle

#close 없이 파일 여는법
with open("profile.pickle", "rb") as profile_file:
    print(pickle.load(profile_file))

#with 함수를 통해 파일을 쓰고 읽는법
with open("study.txt", "w", encoding="utf8") as study_file:
    study_file.write("파이썬을 공부하고 있어요")

with open("study.txt", "r", encoding="utf8") as study_file:
    print(study_file.read())

print()

for i in range(1,6):
    with open(str(i) + "주차.txt", "w", encoding="utf8") as report_file:
        report_file.write("- {0}주차 주간보고-\n부서 :\n이름 :\n업무 요약 :" . format(i))
        #report_file.write(" - "str(i) + "주차 주간보고-\n부서 :\n이름 :\n업무 요약 :")

#클래스
name = "마린"
hp = 40
damage = 5

print("{0} 유닛이 생성되었습니다.".format(name))
print("체력 {0} , 공격력{1}\n".format(hp,damage))

tnak_name = "탱크"
tnak_hp = 150
tnak_damage = 35

print("{0} 유닛이 생성되었습니다.".format(tnak_name))
print("체력 {0} , 공격력{1}\n".format(tnak_hp,tnak_damage))

def attack(name, location, damage):
    print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]".format(\
        name, location, damage))

attack(name, "1시", damage)
attack(tnak_name, "1시", tnak_damage)

print()

class Unit:
    def __init__(self, name, hp, damage): #생성자: 자동으로 호출되는 부분, self를 제외한 다른 부분을 모두 넘겨줘야 객체를 만들 수 있다 
        self.name = name    #맴버 변수
        self.hp = hp
        self.damage = damage
        print("{0} 유닛이 생성되었습니다.".format(self.name))
        print("체력 {0}, 공격력{1}.".format(self.hp, self.damage))

marine1 = Unit("마린", 40, 5) #객체: 클래스로부터 만들어지는 것,(Unit클래스의 인스턴스)
marine2 = Unit("마린", 40, 5)
tank1 = Unit("탱크", 150, 35)
wraith1 = Unit("레이스", 80, 5)
print("유닛 이름 : {0}, 공격력 : {1}".format(wraith1.name, wraith1.damage)) #맴버 변수를 외부에서 사용 가능

wraith2 = Unit(" 빼앗은 레이스", 80, 5)
wraith2.clocking = True #클래스 외부에서 추가로 변수를 만들어 객체에 추가 가능(추가한 객체에서만 적용이 된다)

if wraith2.clocking == True:
    print("{0} 는 현재 클로킹 상태 입니다.".format(wraith2.name))

class AttackUnit:
    def __init__(self, name, hp, damage): #생성자: 자동으로 호출되는 부분, self를 제외한 다른 부분을 모두 넘겨줘야 객체를 만들 수 있다 
        self.name = name    #맴버 변수
        self.hp = hp
        self.damage = damage
    
    def attack(self, location):
        print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]".format(\
        self.name, location, self.damage)) #name 과 damage는 자기 자신의 정보, location은 전달받은 정보

    def damaged(self, damage):
        print("{0} : {1} 데미지를 입었습니다.".format(self.name, damage))
        self.hp -= damage
        print("{0} : 현재 체력 {1} 입니다.".format(self.name, self.hp))
        if self.hp <= 0:
            print("{0} : 파괴되었습니다.".format(self.name))

firebat1 = AttackUnit("파이어벳", 50, 16)
firebat1.attack("5시")
firebat1.damaged(25)
firebat1.damaged(25)

print()

#상속
class Unit: #부모 클래스
    def __init__(self, name, hp): 
        self.name = name  
        self.hp = hp

class AttackUnit(Unit): #자식 클래스 #공격 유닛은 일반 유닛을 상속받음
    def __init__(self, name, hp, damage): #생성자
        Unit.__init__(self, name, hp) #unit에서 만들어진 생성자를 상속받아 이름과 체력 정의 (초기화)
        self.damage = damage
    
    def attack(self, location):
        print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]".format(\
        self.name, location, self.damage))

    def damaged(self, damage):
        print("{0} : {1} 데미지를 입었습니다.".format(self.name, damage))
        self.hp -= damage
        print("{0} : 현재 체력 {1} 입니다.".format(self.name, self.hp))
        if self.hp <= 0:
            print("{0} : 파괴되었습니다.".format(self.name))

firebat1 = AttackUnit("파이어벳", 50, 16)
firebat1.attack("5시")
firebat1.damaged(25)
firebat1.damaged(25)
print()

#다중 상속 #여려 부모에서 상속 받음

#날 수 있는 기능을 가진 클래스
class Flyable:
    def __init__(self, flying_speed):
        self.flying_speed = flying_speed
    
    def fly(self, name, location):
        print("{0} : {1} 방향으로 날아갑니다. [속도 {2}]".format(name, location, self.flying_speed))

#공중 공격 유닛 클래스
class FlyableAttackUnit(AttackUnit,Flyable): #다중 상속
    def __init__(self, name, hp, damage, flying_speed): #생성자
        AttackUnit.__init__(self, name, hp, damage)     #AttackUnit 초기화
        Flyable.__init__(self, flying_speed)            #Flyable 초기화

valkyrie = FlyableAttackUnit("발키리", 200, 6, 5)
valkyrie.fly(valkyrie.name, "3시")
print()
print()




#메소드 오버로딩
class Unit: #일반 유닛
    def __init__(self, name, hp, speed): 
        self.name = name  
        self.hp = hp
        self.speed = speed

    def move(self, location):
        print("[지상 유닛 이동]")
        print("{0} : {1} 방향으로 이동합니다. [속도 {2}]".format(self.name, location, self.speed))

class AttackUnit(Unit): 
    def __init__(self, name, hp, speed, damage): 
        Unit.__init__(self, name, hp, speed) 
        self.damage = damage
    
    def attack(self, location):
        print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]".format(\
        self.name, location, self.damage))

    def damaged(self, damage):
        print("{0} : {1} 데미지를 입었습니다.".format(self.name, damage))
        self.hp -= damage
        print("{0} : 현재 체력 {1} 입니다.".format(self.name, self.hp))
        if self.hp <= 0:
            print("{0} : 파괴되었습니다.".format(self.name))

class FlyableAttackUnit(AttackUnit,Flyable):
    def __init__(self, name, hp, damage, flying_speed): 
        AttackUnit.__init__(self, name, hp, 0, damage)     #지상 스피드 0
        Flyable.__init__(self, flying_speed)            

    def move(self, location): # 메소드 오버로딩 #move 재정의
        print("[공중 유닛 이동]")
        self.fly(self.name, location)

vulture = AttackUnit("벌쳐", 80, 10, 20)
battlecruiser = FlyableAttackUnit("배틀크루저", 500, 25, 3)

vulture.move("11시")
battlecruiser.fly(battlecruiser.name, "9시")
battlecruiser.move("9시") #move 재정의


#pass
#건물
class BuildingUnit(Unit):
    def __init__(self, name, hp, location):
        pass #함수를 완성하지 않아도 실행되지만 아무것도 하지 않음

supply_depot = BuildingUnit("서플라이 디폿", 500, "7시")

print()
#super
class BuildingUnit(Unit):
    def __init__(self, name, hp, location):
        #Unit.__init__(self, name, hp, 0) #건물은 스피드 0
        super().__init__(name, hp, 0) #상속 받아 초기화를 할때 super()을 통해도 가능 ()들어가고 self는 빠짐
        self.location = location


class Unit:
    def __init__(self):
        print("Unit 생성자")

class Flyable:
    def __init__(self):
        print("Flyable 생성자")

class FlyableUnit(Unit, Flyable):
    def __init__(self):
        super().__init__()
        #Unit.__init__(self)
        #FlyableAttackUnit.__init__(self)
dropship = FlyableUnit() # 두개 이상의 다중 상속을 받을 때는 super()을 사용하면 맨 처음 클래스에 대해서만 초기화가 진행된다

print()
print()
#부동산 프로그램
class House:
    def __init__(self, location, house_type, deal_type, price, completion_year):
        self.location = location
        self.house_type = house_type
        self.deal_type = deal_type
        self.price = price
        self.completion_year = completion_year

    def show_detail(self, number):
        if number > 0:
                print("{0} {1} {2} {3} {4}년".format(self.location, self.house_type, self.deal_type, self.price, self.completion_year))
        else:
            print("매물이 없습니다.")

houses = []
house1 = House("강남", "아파트", "매매", "10억", "2010")
house2 = House("마포", "오피스텔", "전세", "5억", "2007")
house3 = House("송파", "빌라", "월세", "500/50", "2000")

houses.append(house1)
houses.append(house2)
houses.append(house3)

print("총 {0}대의 매물이 있습니다.".format(len(houses)))
for house in houses:
    house.show_detail(len(houses))


print()
print()

#예외처리

try: 
    print("나누기 전용 계산기 입니다.")
    nums = []
    # nums.append(int(input("첫 번째 숫자를 입력하세요 : ")))
    # nums.append(int(input("두 번째 숫자를 입력하세요 : ")))
    #nums.append(int(nums[0]/nums[1]))
    print("{0} / {1} = {2}".format(nums[0], nums[1], nums[2]))
except ValueError: #try 내부 문장 오류 발생시 except 부분에 해당하는 오류면 except 내부 문장 출력
    print("에러! 잘못된 값을 입력하였습니다.")
except ZeroDivisionError as err:
    print(err) #발생하는 에러 문장을 출력
except Exception as err: #나머지 모든 오류
    print("알 수 없는 에러가 발생하였습니다.")
    print(err) #발생하는 에러 문장을 출력

print()

#원하지 않는 조건 에러 만들기

# try: 
#     print("한 자리 숫자 나누기 전용 계산기입니다.")

#     num1 = int(input("첫 번째 숫자를 입력하세요 : "))
#     num2 = int(input("두 번째 숫자를 입력하세요 : "))
#     if num1 >= 10 or num2 >=10:
#         raise ValueError #원하지 않는 조건 에러 만들기 -> ValueError
#     print("{0} / {1} = {2}".format(num1, num2, int(num1/num2)))
# except ValueError: # ->ValueError 처리 구문
#     print("잘못된 값을 입력하였습니다. 한 자리 숫자만 입력하세요.")

print()

#사용자 정의 예외처리
class BignNumberError(Exception):
    def __init__(self, msg):
        self.msg = msg
    
    def __str__(self):
        return self.msg


# try: 
#     print("한 자리 숫자 나누기 전용 계산기입니다.")
#     num1 = int(input("첫 번째 숫자를 입력하세요 : "))
#     num2 = int(input("두 번째 숫자를 입력하세요 : "))
#     if num1 >= 10 or num2 >=10:
#         raise BignNumberError("입력값 : {0}, {1}".format(num1,num2)) #사용자 정의 조건에 의해 오류가 발생하면 특정 문자열 넘김
#     print("{0} / {1} = {2}".format(num1, num2, int(num1/num2)))
# except ValueError: 
#     print("잘못된 값을 입력하였습니다. 한 자리 숫자만 입력하세요.")
# except BignNumberError as err: 
#     print("에러가 발생하였습니다. 한 자리 숫자만 입력하세요.")
#     print(err) #넘긴 문자열을 출력함
# finally: #finally #오류가 있든 없든, 정의한 오류건 아니건 무조건 발생하는 문장, 프로그램이 강제종류를 막고 완성도를 높힘
#     print("계산기를 이용해 주셔서 감사합니다.")


#자동주문 시스템
# chicken = 10
# waiting = 1
# class SoldOutError(Exception):
#     def __init__(self, msg):
#         self.msg = msg

#     def __str__(self):
#         return self.msg

# while(True):
#     try:
#         print("[남은 치킨 : {0}]".format(chicken))
#         order = int(input("치킨 몇마리 주문하시겠습니까"))
#         if order < 1:
#             raise ValueError
#         elif order > chicken:
#             print("재료가 부족합니다.")
#         else:
#             print("[대기번호 {0}] {1} 마리 주문이 완료되었습니다.".format(waiting, order))
#             waiting += 1
#             chicken -= order
#         if chicken == 0:
#             raise SoldOutError("재고가 소진되어 더 이상 주문을 받지 않습니다.")
#     except ValueError:
#         print("잘못된 값을 입력하였습니다.")
#     except SoldOutError as err:
#         print(err) 
#         break
#     except Exception as err:
#         print(err)


print()

#모듈화
import theater_module #[import 모듈]
theater_module.price(3)
theater_module.price_morning(4)
theater_module.price_soldier(5)

import theater_module as mv #theater_module 의 이름을  mv 로 부른다 [import 모듈 as 별명]
mv.price(3)
mv.price_morning(4)
mv.price_soldier(5)

from theater_module import * #모듈 이름을 쓸 필요가 없이 함수를 가져다 쓸 수 있음. [from 모듈 import *]
price(3)
price_morning(4)
price_soldier(5)

from theater_module import price, price_morning #필요한 함수만 가져다 쓸 수 있다. [from 모듈 import 함수, 함수]
price(3)
price_morning(4)
price_soldier(5) # 사용 불가능하다

from theater_module import price_soldier as price #price_soldier를 price별명을 붙여쓴다 [from 모듈 import 함수 as 별명]
price(5) #price_soldier  함수의 별명

print()

# #패키지 -> 모듈들의 집합, 하나의 디렉코리에 여러 모듈파일들을 넣어놓은 것
# import travel.thailand #패키지 로드 #impor구문에서는 [import 패키지.모듈] 만 사용가능
# trip_to = travel.thailand.ThailandPackage()
# trip_to.detail()

# from travel.thailand import ThailandPackage #from impor 구문에서는 [from 패키지.모듈 import 클래스]
# trip_to = ThailandPackage()
# trip_to.detail()

# from travel import vietnam #[from 패키지 import 모듈]
# trip_to = vietnam.VietnamPackage()
# trip_to.detail() 

from travel import * # *은 travel 패키지의 모든것을 가져오겠다는 것이지만 개발자가 공개범위를 설정해야함 #__all__ = ["vietnam"]
trip_to = vietnam.VietnamPackage()
trip_to.detail() 

#모듈 직접 실행 외부 실행
trip_to = thailand.ThailandPackage()
trip_to.detail() 

#패키지 모듈 위치
import inspect
import random
print(inspect.getfile(random))
print(inspect.getfile(thailand))

#출력창 지우기 -> cls


#pip로 패키지 설치하기
#pip list -> 설치된 패키지 보기
#pip show 패키지 이름 -> 패키지 정보
#pip install --upgrade 패키지 이름 -> 패키지 최신버전으로 설치
#pip uninstall 패키지명 -> 패키지 삭제


#내장함수: 내장되어있어 따로 import 하지 않아도 되는 함수

#input: 사용자 입력을 문자열로 받는 함수
#dir : 어떤 객체를 넘겨줬을 때 그 객체가 어떤 변수와 함수를 가지고 있는지 표시
print(dir())
print()

import random 
print(dir()) #현재 import된 외부 모듈들을 보여줌
print()

print(dir(random)) #random 모듈내 사용가능한 함수들을 보여줌
print()

lst = [1,2,3]
print(dir(lst)) #list에서 사용가능한 함수
print()

name = "Jim"
print(dir(name)) #문자열에서 사용가능한 함수
print()


#list of python builtins 내장함수 검색 
#list of python modules 외장함수(모듈) 검색

# glob : 경로 내의 폴더/ 파일 목록 조회 (윈도우의 dir 과 같다.)
import glob
print(glob.glob("*.py")) #경로 내 확장자가 py인 모든 파일
print()

#os : 운영체제에서 제공하는 기본 기능
import os
print(os.getcwd()) #현재 디렉토리 출력
print()

folder = "sample_dir" #폴더 생성 삭제
if os.path.exists(folder): #경로에 폴더가 존재하는지 아닌지 확인
    print("이미 존재하는 폴더입니다.")
    os.rmdir(folder) #폴더 지움
    print(folder, "폴더를 삭제하였습니다.")
else:
    os.makedirs(folder) #폴더 생성
    print(folder, "폴더를 생성하였습니다.")

print()
print(os.listdir()) #디렉토리에 존재하는 파일

print()
import time #시간 관련 함수
print(time.localtime())
print(time.strftime("%y-%m-%d %H:%M:%S")) #원하는 정보 출력 가능

print()
import datetime
print("오늘 날짜는 ", datetime.date.today())

print()
#timedelta : 두 날짜 사이의 간격
today = datetime.date.today() #오늘 날짜 저장
td = datetime.timedelta(days=100) #100일 저장 
print("우리가 만난지 100일은", today + td) #오늘부터 100일 후
print()

import byme
byme.sign()

# 파이썬 기초 끝