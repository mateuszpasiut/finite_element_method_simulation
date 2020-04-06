import math
import numpy as np
from integralPoint import IntegralPoint

class UniversalElement:
    def __init__(self):
        self.num_ip = 4
        self.num_points = 4
        self.num_shape_f = 4
        self.dN_dKsi = [[0] * 4, [0] * 4, [0] * 4, [0] * 4]
        self.dN_dEta = [[0] * 4, [0] * 4, [0] * 4, [0] * 4]
        self.N = [[0] * 4, [0] * 4, [0] * 4, [0] * 4]

        self.dN_dKsi_dN_dEta = np.zeros((self.num_points, self.num_shape_f, 2, 1))

        self.integral_points = [
            IntegralPoint(-1 / math.sqrt(3), -1 / math.sqrt(3)),
            IntegralPoint(1 / math.sqrt(3), -1 / math.sqrt(3)),
            IntegralPoint(1 / math.sqrt(3), 1 / math.sqrt(3)),
            IntegralPoint(-1 / math.sqrt(3), 1 / math.sqrt(3))
        ]

        self.integral_points_hbc = [
            [IntegralPoint(-1 / math.sqrt(3), -1), IntegralPoint(1 / math.sqrt(3), -1)],
            [IntegralPoint(1, -1 / math.sqrt(3)), IntegralPoint(1, 1 / math.sqrt(3))],
            [IntegralPoint(1 / math.sqrt(3), 1), IntegralPoint(-1 / math.sqrt(3), 1)],
            [IntegralPoint(-1, 1 / math.sqrt(3)), IntegralPoint(-1, -1 / math.sqrt(3))]
        ]

    def fill_dN_dKsi_matrix(self):
        for i in range(self.num_points):
            self.dN_dKsi[i][0] = (-0.25 * (1 - self.integral_points[i].eta))
            self.dN_dKsi[i][1] = (0.25 * (1 - self.integral_points[i].eta))
            self.dN_dKsi[i][2] = (0.25 * (1 + self.integral_points[i].eta))
            self.dN_dKsi[i][3] = (-0.25 * (1 + self.integral_points[i].eta))

    def fill_dN_dEta_matrix(self):
        for i in range(self.num_points):
            self.dN_dEta[i][0] = (-0.25 * (1 - self.integral_points[i].ksi))
            self.dN_dEta[i][1] = (-0.25 * (1 + self.integral_points[i].ksi))
            self.dN_dEta[i][2] = (0.25 * (1 + self.integral_points[i].ksi))
            self.dN_dEta[i][3] = (0.25 * (1 - self.integral_points[i].ksi))

    def fill_N_matrix(self):
        for i in range(self.num_points):
            self.N[i][0] = (0.25 * (1 - self.integral_points[i].ksi) * (1 - self.integral_points[i].eta))
            self.N[i][1] = (0.25 * (1 + self.integral_points[i].ksi) * (1 - self.integral_points[i].eta))
            self.N[i][2] = (0.25 * (1 + self.integral_points[i].ksi) * (1 + self.integral_points[i].eta))
            self.N[i][3] = (0.25 * (1 - self.integral_points[i].ksi) * (1 + self.integral_points[i].eta))

    def fill_dN_dKsi_dN_dEta_matrix(self):
        for i in range(self.num_points):
            for j in range(self.num_shape_f):
                self.dN_dKsi_dN_dEta[i][j][0] = self.dN_dKsi[i][j]
                self.dN_dKsi_dN_dEta[i][j][1] = self.dN_dEta[i][j]

    def display_matrix(self):
        for i in range(self.num_points):
            print(self.N[i][0], ' ', self.N[i][1], ' ', self.N[i][2], ' ', self.N[i][3])

    def display(self):
        for i in range(self.num_points):
            print(self.dN_dKsi[i][0], ' ', self.dN_dKsi[i][1], ' ', self.dN_dKsi[i][2], ' ', self.dN_dKsi[i][3], ' ',)



