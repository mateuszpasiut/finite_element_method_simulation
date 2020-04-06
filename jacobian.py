class Jacobian:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.jacobian = [[0] * self.nx, [0] * self.ny]

    def display(self):
        print(self.jacobian[0][0], self.jacobian[0][1])
        print(self.jacobian[1][0], self.jacobian[1][1])
        print("\n")
