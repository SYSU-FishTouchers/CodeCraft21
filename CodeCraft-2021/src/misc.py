import numpy as np
from numpy.linalg import inv

class PlaneModel:
    def __init__(self, xs, ys, zs):
        assert len(xs) == len(ys) == len(zs)

        N = len(xs)

        # 创建系数矩阵A
        A = np.zeros((3, 3))
        for x, y in zip(xs, ys):
            A[0, 0] = A[0, 0] + x ** 2
            A[1, 0] = A[0, 1] = A[0, 1] + x * y
            A[2, 0] = A[0, 2] = A[0, 2] + x
            A[1, 1] = A[1, 1] + y ** 2
            A[2, 1] = A[1, 2] = A[1, 2] + y
            A[2, 2] = N

        # 创建b
        b = np.zeros((3, 1))
        for x, y, fixed_cost in zip(xs, ys, zs):
            b[0, 0] = b[0, 0] + x * fixed_cost
            b[1, 0] = b[1, 0] + y * fixed_cost
            b[2, 0] = b[2, 0] + fixed_cost

        self.a1, self.a2, self.a3 = np.dot(inv(A), b)[:3, 0]

    def predict(self, x, y):
        return self.a1 * x + self.a2 * y + self.a3
