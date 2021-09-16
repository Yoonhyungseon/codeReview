class Unit:
    def __init__(self):
        print("Unit 생성자")

class Flyable:
    def __init__(self):
        print("Flyable 생성자")

class FlyableUnit(Unit, Flyable):
    def __init__(self):
        super().__init__()

dropship = FlyableUnit() # 두개 이상의 다중 상속을 받을 때는 super()을 사용하면 맨 처음 클래스에 대해서만 초기화가 진행된다