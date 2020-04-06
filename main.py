import globalData
from grid import Grid
import universalElement
import jacobian
import numpy as np
import os


class Start:
    def __init__(self, plik):
        self.dane = []
        self.read_from_file(plik)

        self.h = float(self.dane[0])
        self.w = float(self.dane[1])
        self.nH = int(self.dane[2])
        self.nW = int(self.dane[3])
        self.initial_temperature = float(self.dane[4])
        self.simulation_time = float(self.dane[5])
        self.simulation_step_time = float(self.dane[6])
        self.ambient_temperature = float(self.dane[7])
        self.alfa = float(self.dane[8])
        self.specific_heat = float(self.dane[9])
        self.conductivity = float(self.dane[10])
        self.desity = float(self.dane[11])
        self.number_of_steps = int(self.simulation_time / self.simulation_step_time)
        self.hbc_weights = [1, 1]
        self.weights = [1, 1, 1, 1]
        self.nSides = 4
        self.n_bc_points = 2

        self.start = globalData.GlobalData(self.h, self.w, self.nH, self.nW)

        self.grid = Grid()

        self.start.fill_nodes(self.grid)
        self.start.fill_elements(self.grid)

        self.ue = universalElement.UniversalElement()

        self.ue.fill_dN_dKsi_matrix()
        self.ue.fill_dN_dEta_matrix()
        self.ue.fill_N_matrix()
        self.ue.fill_dN_dKsi_dN_dEta_matrix()

        self.jacobian = jacobian.Jacobian(2, 2)
        self.jacobians = [[self.jacobian] * self.ue.num_points] * self.start.nE
        self.inverse_of_jacobians = [[self.jacobian] * self.ue.num_points] * self.start.nE

        self.h = np.zeros((self.start.nE, self.ue.num_points, self.ue.num_shape_f))
        self.c = np.zeros((self.start.nE, self.ue.num_points, self.ue.num_shape_f))
        self.hbc = np.zeros((self.nSides, self.ue.num_shape_f, self.ue.num_shape_f))
        self.hbc1 = np.zeros((self.nSides, self.n_bc_points, self.ue.num_shape_f, 1))
        self.hc_dt = np.zeros((self.start.nN, self.start.nN))

        self.dN_dx_dN_y = np.zeros((self.start.nE, 4, 4, 2, 1))
        self.h_matrix = np.zeros((self.start.nE, self.ue.num_points, self.ue.num_shape_f, self.ue.num_shape_f))
        self.hhbc = np.zeros((self.start.nE, self.ue.num_shape_f, self.ue.num_shape_f))

        self.hg = np.zeros((self.start.nN, self.start.nN))
        self.cg = np.zeros((self.start.nN, self.start.nN))

        self.p = np.zeros((self.nSides, self.ue.num_shape_f, 1))
        self.p_e = np.zeros((self.start.nE, self.ue.num_shape_f, 1))
        self.pg = np.zeros((self.start.nN, 1))

        self.t1 = np.zeros((self.start.nN, 1))
        self.p_t0 = self.pg + (self.cg/self.simulation_step_time).dot(self.t1)
        self.min_max_temperatures = np.zeros((self.number_of_steps, 1, 2))

        self.calculate_jacobians()
        self.calculate_inverse_of_jacobians()

        self.calculate_dN_dx_dN_dy()

        self.calculate_h_matrix()
        self.calculate_c_matrix()

        self.calculate_hbc()
        self.calculate_hhbc()

        self.calculate_p()
        self.calculate_pg()

        self.matrix_aggregation()
        self.calculate_hc_dt()

        self.t1.fill(100)

        self.mes()

    def calculate_jacobians(self):
        for i in range(self.start.nE):
            for j in range(self.ue.num_points):
                jacobi = jacobian.Jacobian(2, 2)
                for k in range(self.ue.num_shape_f):
                    jacobi.jacobian[0][0] += self.ue.dN_dKsi[j][k] * self.grid.nodes[self.grid.elements[i].nodes[k]-1].x
                    jacobi.jacobian[0][1] += self.ue.dN_dKsi[j][k] * self.grid.nodes[self.grid.elements[i].nodes[k]-1].y
                    jacobi.jacobian[1][0] += self.ue.dN_dEta[j][k] * self.grid.nodes[self.grid.elements[i].nodes[k]-1].x
                    jacobi.jacobian[1][1] += self.ue.dN_dEta[j][k] * self.grid.nodes[self.grid.elements[i].nodes[k]-1].y
                self.jacobians[i][j] = jacobi

    def calculate_inverse_of_jacobians(self):
        for i in range(self.start.nE):
            for j in range(self.ue.num_points):
                self.inverse_of_jacobians[i][j] = np.linalg.inv(self.jacobians[i][j].jacobian)

    def calculate_dN_dx_dN_dy(self):
        for i in range(self.start.nE):
            for j in range(self.ue.num_points):
                for k in range(self.ue.num_shape_f):
                    self.dN_dx_dN_y[i][j][k] = self.inverse_of_jacobians[i][k].dot(self.ue.dN_dKsi_dN_dEta[j][k])

    def calculate_h_matrix(self):
        h = np.zeros((self.start.nE, 4, 2, 4, 1))
        for i in range(self.start.nE):
            final_h = np.zeros((self.ue.num_shape_f, self.ue.num_shape_f))
            for j in range(self.ue.num_points):
                for k in range(2):
                    for l in range(self.ue.num_shape_f):
                        h[i][j][k][l] = self.dN_dx_dN_y[i][j][l][k]
                self.h_matrix[i][j] = h[i][j][0].dot(h[i][j][0].transpose()) + h[i][j][1].dot(h[i][j][1].transpose())
                final_h += self.h_matrix[i][j] * self.weights[j] * np.linalg.det(self.jacobians[i][j].jacobian)
                self.h[i] = final_h * self.conductivity

    def calculate_c_matrix(self):
        c1 = np.zeros((self.ue.num_points, self.ue.num_shape_f, 1))
        for i in range(self.ue.num_points):
            for j in range(self.ue.num_shape_f):
                c1[i][j] = self.ue.N[i][j]

        c2 = np.zeros((self.start.nE, self.ue.num_points, self.ue.num_shape_f, self.ue.num_shape_f))
        for i in range(self.start.nE):
            for j in range(self.ue.num_points):
                c2[i][j] = c1[j].dot(c1[j].transpose()) * np.linalg.det(self.jacobians[i][j].jacobian) * self.specific_heat * self.desity
                self.c[i] += c2[i][j]

    def calculate_hbc(self):

        self.hbc1[0][0][0] = self.calculate_hbc_s(1, 0, 0)
        self.hbc1[0][0][1] = self.calculate_hbc_s(2, 0, 0)
        self.hbc1[0][1][0] = self.calculate_hbc_s(1, 0, 1)
        self.hbc1[0][1][1] = self.calculate_hbc_s(2, 0, 1)

        self.hbc1[1][0][1] = self.calculate_hbc_s(2, 1, 0)
        self.hbc1[1][0][2] = self.calculate_hbc_s(3, 1, 0)
        self.hbc1[1][1][1] = self.calculate_hbc_s(2, 1, 1)
        self.hbc1[1][1][2] = self.calculate_hbc_s(3, 1, 1)

        self.hbc1[2][0][2] = self.calculate_hbc_s(3, 2, 0)
        self.hbc1[2][0][3] = self.calculate_hbc_s(4, 2, 0)
        self.hbc1[2][1][2] = self.calculate_hbc_s(3, 2, 1)
        self.hbc1[2][1][3] = self.calculate_hbc_s(4, 2, 1)

        self.hbc1[3][0][0] = self.calculate_hbc_s(1, 3, 0)
        self.hbc1[3][0][3] = self.calculate_hbc_s(4, 3, 0)
        self.hbc1[3][1][0] = self.calculate_hbc_s(1, 3, 1)
        self.hbc1[3][1][3] = self.calculate_hbc_s(4, 3, 1)

        for i in range(0, 4, 2):
            self.hbc[i] = (self.hbc1[i][0].dot(self.hbc1[i][0].transpose()) * self.hbc_weights[0] + self.hbc1[i][1].dot(self.hbc1[i][1].transpose()) * self.hbc_weights[1]) * self.alfa * self.start.dx / 2
            self.hbc[i+1] = (self.hbc1[i+1][0].dot(self.hbc1[i+1][0].transpose()) * self.hbc_weights[0] + self.hbc1[i+1][1].dot(self.hbc1[i+1][1].transpose()) * self.hbc_weights[1]) * self.alfa * self.start.dy / 2

    def calculate_hbc_s(self, fn, i, j):
        if fn == 1:
            return 0.25 * (1 - self.ue.integral_points_hbc[i][j].ksi) * (1 - self.ue.integral_points_hbc[i][j].eta)
        if fn == 2:
            return 0.25 * (1 + self.ue.integral_points_hbc[i][j].ksi) * (1 - self.ue.integral_points_hbc[i][j].eta)
        if fn == 3:
            return 0.25 * (1 + self.ue.integral_points_hbc[i][j].ksi) * (1 + self.ue.integral_points_hbc[i][j].eta)
        if fn == 4:
            return 0.25 * (1 - self.ue.integral_points_hbc[i][j].ksi) * (1 + self.ue.integral_points_hbc[i][j].eta)

    def calculate_hhbc(self):
        for i in range(self.start.nE):
            self.hhbc[i] = np.copy(self.h[i])

        j = self.start.nE - 1
        for i in range(self.start.nH - 1):
            self.hhbc[i] += self.hbc[3]
            self.hhbc[j] += self.hbc[1]
            j -= 1

        j = self.start.nH - 2
        i = 0
        while i < self.start.nE - 1:
            self.hhbc[i] += self.hbc[0]
            self.hhbc[j] += self.hbc[2]
            i += self.start.nH - 1
            j += self.start.nH - 1

    def matrix_aggregation(self):
        for i in range(self.start.nE):
            i1 = 0
            for j in self.grid.elements[i].nodes:
                i2 = 0
                for k in self.grid.elements[i].nodes:
                    self.hg[j-1][k-1] += self.hhbc[i][i1][i2]
                    self.cg[j-1][k-1] += self.c[i][i1][i2]
                    i2 += 1
                i1 += 1

    def calculate_hc_dt(self):
        self.hc_dt = self.hg + self.cg / self.simulation_step_time

    def calculate_p(self):
        p = np.zeros((self.nSides, self.n_bc_points, self.ue.num_shape_f, 1))

        p[0][0][0] = -self.calculate_p_s(1, 0, 0)
        p[0][0][1] = -self.calculate_p_s(2, 0, 0)
        p[0][1][0] = -self.calculate_p_s(1, 0, 1)
        p[0][1][1] = -self.calculate_p_s(2, 0, 1)

        p[1][0][1] = -self.calculate_p_s(2, 1, 0)
        p[1][0][2] = -self.calculate_p_s(3, 1, 0)
        p[1][1][1] = -self.calculate_p_s(2, 1, 1)
        p[1][1][2] = -self.calculate_p_s(3, 1, 1)

        p[2][0][2] = -self.calculate_p_s(3, 2, 0)
        p[2][0][3] = -self.calculate_p_s(4, 2, 0)
        p[2][1][2] = -self.calculate_p_s(3, 2, 1)
        p[2][1][3] = -self.calculate_p_s(4, 2, 1)

        p[3][0][0] = -self.calculate_p_s(1, 3, 0)
        p[3][0][3] = -self.calculate_p_s(4, 3, 0)
        p[3][1][0] = -self.calculate_p_s(1, 3, 1)
        p[3][1][3] = -self.calculate_p_s(4, 3, 1)

        for i in range(0, 4, 2):
            self.p[i] = (self.hbc1[i][0] * self.hbc_weights[0] + self.hbc1[i][1] * self.hbc_weights[1]) * self.alfa * self.ambient_temperature * self.start.dx / 2
            self.p[i+1] = (self.hbc1[i+1][0] * self.hbc_weights[0] + self.hbc1[i+1][1] * self.hbc_weights[1]) * self.alfa * self.ambient_temperature * self.start.dy / 2

        j = self.start.nE - 1
        for i in range(self.start.nH - 1):
            self.p_e[i] += self.p[3]
            self.p_e[j] += self.p[1]
            j -= 1

        j = self.start.nH - 2
        i = 0
        while i < self.start.nE - 1:
            self.p_e[i] += self.p[0]
            self.p_e[j] += self.p[2]
            i += self.start.nH - 1
            j += self.start.nH - 1

    def calculate_p_s(self, fn, i, j):
        if fn == 1:
            return 0.25 * (1 - self.ue.integral_points_hbc[i][j].ksi) * (1 - self.ue.integral_points_hbc[i][j].eta)
        if fn == 2:
            return 0.25 * (1 + self.ue.integral_points_hbc[i][j].ksi) * (1 - self.ue.integral_points_hbc[i][j].eta)
        if fn == 3:
            return 0.25 * (1 + self.ue.integral_points_hbc[i][j].ksi) * (1 + self.ue.integral_points_hbc[i][j].eta)
        if fn == 4:
            return 0.25 * (1 - self.ue.integral_points_hbc[i][j].ksi) * (1 + self.ue.integral_points_hbc[i][j].eta)

    def calculate_pg(self):
        for i in range(self.start.nE):
            i1 = 0
            for j in self.grid.elements[i].nodes:
                self.pg[j-1] += self.p_e[i][i1]
                i1 += 1

    def mes(self):
        for i in range(self.number_of_steps):
            self.t1 = np.linalg.inv(self.hc_dt).dot((self.cg/self.simulation_step_time).dot(self.t1) + self.pg)
            self.min_max_temperatures[i] = [self.t1.min(), self.t1.max()]
            print(self.min_max_temperatures[i])

    def read_from_file(self, data):
        if os.path.isfile(data):
            with open(data, "r") as z:
                for line in z:
                    self.dane.append(line)


if __name__ == '__main__':
    s1 = Start("data.txt")
