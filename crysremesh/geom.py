from numpy import sqrt
from numpy import array


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def coords(self):
        return self.x, self.y, self.z


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def coords(self):
        return self.x, self.y, self.z

    def coords_np_array(self):
        return array([self.x, self.y, self.z]).reshape((1, 3))

    def norm(self):
        return euclidian_distance(self.point(), Point(0, 0, 0))

    def point(self):
        return Point(self.x, self.y, self.z)

    def sum(self, other):
        assert isinstance(other, Vector)
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def sub(self, other):
        assert isinstance(other, Vector)
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def dev(self, k):
        if k == 0:
            raise ZeroDivisionError
        self.mul(1 / k)

    def make_unit(self):
        l = self.norm()
        self.dev(l)

    def mul(self, k):
        self.x *= k
        self.y *= k
        self.z *= k

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

    @staticmethod
    def subtract_vectors(v1, v2):
        return Vector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)


def edge_to_vector(e) -> Vector:
    from .triangular_grid import Edge
    assert len(e.nodes) == 2, 'Wrong number of nodes in the edge'
    n1 = e.nodes[0]
    n2 = e.nodes[1]
    p1 = point_to_vector(n1.as_point())
    p2 = point_to_vector(n2.as_point())
    return Vector.subtract_vectors(p2, p1)


def dot_product(v1: Vector, v2: Vector) -> float:
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def cross_product(v1: Vector, v2: Vector):
    return Vector(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x)


def point_to_vector(point: Point):
    return Vector(point.x, point.y, point.z)


def euclidian_distance(p1: Point, p2: Point):
    """:param n1, n2 Nodes"""
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
