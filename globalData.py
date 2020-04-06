from node import Node
from grid import Grid
from element import Element


class GlobalData:
    def __init__(self, h, w, nH, nW):
        self.h = h
        self.w = w
        self.nH = nH
        self.nW = nW
        self.nN = nH * nW
        self.nE = (nH-1)*(nW-1)
        self.dx = self.w/(self.nW-1)
        self.dy = self.h/(self.nH-1)

    def fill_nodes(self, Grid):
        x = 0
        y = 0
        id = 1
        i = 0

        while i < self.nW:
            j = 0
            i += 1
            while j < self.nH:
                j += 1
                node = Node(x, y, id)

                if x == 0 or i % self.nW == 0:
                    node.bc = True
                if y == 0 or j % self.nH == 0:
                    node.bc = True

                Grid.nodes.append(node)

                y = y + self.dy
                id += 1

            y = 0
            x = x + self.dx

    def fill_elements(self, Grid):
        i = 0
        tmp = 0

        while i < self.nE:
            tmp = tmp + 1
            if tmp % self.nH != 0:
                    i = i + 1
                    n1 = tmp
                    n2 = self.nH + tmp
                    n3 = self.nH + tmp + 1
                    n4 = tmp + 1
                    element = Element(i, n1, n2, n3, n4)
                    Grid.elements.append(element)

