class Element:
    def __init__(self, id, n1, n2, n3, n4):
        self.id = id
        self.nodes = [n1, n2, n3, n4]
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4

    def display(self):
        print(self.id)
        print(self.n1)
        print(self.n2)
        print(self.n3)
        print(self.n4)
        print('\n')
