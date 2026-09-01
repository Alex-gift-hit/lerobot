class base:
    base: str

    def __init__(self):
        self.base = "base"


class robot(base):
    def __init__(self):
        super().__init__()
        self.robot = "robot"


r = robot()
print(r.base)
print(r.robot)
