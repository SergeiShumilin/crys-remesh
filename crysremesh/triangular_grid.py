from numpy import array, inf, zeros
from .geom import cross_product, point_to_vector, edge_to_vector
from .geom import Vector
from .geom import Point

NODE_COMPARE_ACCURACY = 10e-20


class Edge:
    __doc__ = "Module describing triangular_grid2's edge."

    def __init__(self):
        """
        Construct an edge.
        :param id: edge's id.
        """
        self.Id = None
        self.nodes = list()
        self.faces = list()
        self.border = False


class Face:
    __doc__ = "Module describing triangular_grid2's face."

    def __init__(self, Id=None):
        """
        Construct a face.
        :param id: face's id.
        """
        self.Id = Id

        # Nodes and edges set clockwise.
        self.nodes = list()
        self.nodes_ids = list()
        self.edges = list()

        self.T = None
        self.Hw = None
        self.Hi = None
        self.HTC = None
        self.Beta = None
        self.TauX = None
        self.TauY = None
        self.TauZ = None
        self.aux_node = Node()
        self.vector_median = None
        self.fuzzy_median = Vector()

    def normal(self):
        assert len(self.nodes) == 3, 'Wrong number of nodes in the face'
        p1 = self.nodes[0].as_point()
        p2 = self.nodes[1].as_point()
        p3 = self.nodes[2].as_point()
        v1 = Vector(p1.x, p1.y, p1.z)
        v2 = Vector(p2.x, p2.y, p2.z)
        v3 = Vector(p3.x, p3.y, p3.z)
        vf = Vector.subtract_vectors(v1, v2)
        vs = Vector.subtract_vectors(v1, v3)
        vr = cross_product(vf, vs)
        vr.make_unit()
        assert isinstance(vr, Vector)
        vr.mul(-1)
        return vr

    def area(self):
        """Calculate the area of the face."""
        assert len(self.nodes) == 3, "Wrong number of nodes in the face"
        n1, n2, n3 = self.nodes[0], self.nodes[1], self.nodes[2]
        p1, p2, p3 = n1.as_point(), n2.as_point(), n3.as_point()
        v1 = Vector.subtract_vectors(point_to_vector(p1), (point_to_vector(p2)))
        v2 = Vector.subtract_vectors(point_to_vector(p1), (point_to_vector(p3)))
        res = cross_product(v1, v2).norm() / 2
        assert isinstance(res, float), print('Wrong area', res)
        return res

    def centroid(self):
        n1 = self.nodes[0]
        n2 = self.nodes[1]
        n3 = self.nodes[2]
        x = (n1.x + n2.x + n3.x) / 3
        y = (n1.y + n2.y + n3.y) / 3
        z = (n1.z + n2.z + n3.z) / 3

        return Point(x, y, z)

    def adjacent_faces(self):
        adj_faces = []
        for e in self.edges:
            assert len(e.faces) <= 2
            if e.faces[0] == self:
                if len(e.faces) == 2:
                    adj_faces.append(e.faces[1])
            else:
                assert e.faces[0] is not e.faces[1]
                adj_faces.append(e.faces[0])
        return adj_faces

    def alpha_quality_measure(self):
        """
        Daniel S.H.Lo Finite element mesh generation 2015 p.334
        """
        assert len(self.edges) == 3
        e1 = edge_to_vector(self.edges[0])
        e2 = edge_to_vector(self.edges[1])
        e3 = edge_to_vector(self.edges[2])
        l1 = e1.norm()
        l2 = e2.norm()
        l3 = e3.norm()
        return (4 * (3 ** 0.5) * self.area()) / (l1 ** 2 + l2 ** 2 + l3 ** 2)


class Node:
    __doc__ = "class describing node"

    def __init__(self, x=None, y=None, z=None, Id=None):
        """
        Construct node.
        :param id: node's id in the triangular_grid2.
        """
        self.x = x
        self.y = y
        self.z = z
        self.Id = Id

        self.T = None
        self.Hw = None

        self.faces = list()
        self.edges = list()

        self.fixed = False
        self.component = None

    def coordinates(self) -> tuple:
        return self.x, self.y, self.z

    def as_point(self):
        return Point(self.x, self.y, self.z)

    def move(self, v):
        self.x += v.x
        self.y += v.y
        self.z += v.z


class AVLTreeNode:
    def __init__(self, key=None):
        self.left = None
        self.right = None
        self.key = key
        self.parent = None
        self.balance = 0
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root = None
        self.n = 0

    def insert(self, key):
        self.root = self._insert(self.root, key)
        self.n += 1

    def _insert(self, root, key):
        if not root:
            return AVLTreeNode(key)

        if self.is_node_less(key, root.key):
            left_sub_root = self._insert(root.left, key)
            root.left = left_sub_root
            left_sub_root.parent = root
        else:
            right_sub_root = self._insert(root.right, key)
            root.right = right_sub_root
            right_sub_root.parent = root

        root.height = max(self._get_height(root.left), self._get_height(root.right)) + 1
        root.balance = self._get_height(root.left) - self._get_height(root.right)

        return self.rebalance(root)

    def find(self, key):
        """Find the key in the tree.
        :param key: Node the node to find"""
        found = self._find(self.root, key)
        if not found:
            return None
        else:
            return found.key

    def _find(self, root: AVLTreeNode, key: Node) -> (AVLTreeNode, None):
        """Find key in the tree."""
        if not root:
            return None

        if self.is_node_less(key, root.key):
            return self._find(root.left, key)
        elif self.is_node_less(root.key, key):
            return self._find(root.right, key)
        else:
            return root

    @staticmethod
    def _get_height(root: AVLTreeNode) -> int:
        if not root:
            return 0
        return root.height

    def rebalance(self, root: AVLTreeNode) -> AVLTreeNode:
        if root.balance == 2:
            if root.left.balance < 0:  # L-R case
                root.left = self.rotate_left(root.left)
                return self.rotate_right(root)
            else:  # L-L case
                return self.rotate_right(root)

        elif root.balance == -2:
            if root.right.balance > 0:  # R-L case
                root.right = self.rotate_right(root.right)
                return self.rotate_left(root)
            else:  # R-R case
                return self.rotate_left(root)
        else:
            return root

    def rotate_right(self, root: AVLTreeNode) -> AVLTreeNode:
        pivot = root.left  # set up pointers
        tmp = pivot.right

        pivot.right = root
        pivot.parent = root.parent  # pivot's parent now root's parent
        root.parent = pivot  # root's parent now pivot

        root.left = tmp
        if tmp:  # tmp can be null
            tmp.parent = root

        # update pivot's parent (manually check which one matches the root that was passed)
        if pivot.parent:
            if pivot.parent.left == root:  # if the parent's left subtree is the one to be updated
                pivot.parent.left = pivot  # assign the pivot as the new child
            else:
                pivot.parent.right = pivot  # vice-versa for right child

        # update heights and balance's using tracked heights
        root.height = max(self._get_height(root.left), self._get_height(root.right)) + 1
        root.balance = self._get_height(root.left) - self._get_height(root.right)
        pivot.height = max(self._get_height(pivot.left), self._get_height(pivot.right)) + 1
        pivot.balance = self._get_height(pivot.left) - self._get_height(pivot.right)
        return pivot

    def rotate_left(self, root: AVLTreeNode) -> AVLTreeNode:
        pivot = root.right
        tmp = pivot.left

        pivot.left = root
        pivot.parent = root.parent
        root.parent = pivot

        root.right = tmp
        if tmp:
            tmp.parent = root

        if pivot.parent:
            if pivot.parent.left == root:
                pivot.parent.left = pivot
            else:
                pivot.parent.right = pivot

        root.height = max(self._get_height(root.left), self._get_height(root.right)) + 1
        root.balance = self._get_height(root.left) - self._get_height(root.right)
        pivot.height = max(self._get_height(pivot.left), self._get_height(pivot.right)) + 1
        pivot.balance = self._get_height(pivot.left) - self._get_height(pivot.right)
        return pivot

    @staticmethod
    def is_node_less(n1: Node, n2: Node) -> bool:
        """Is n1 < n2 coordinate-wise."""
        if abs(n1.x - n2.x) > NODE_COMPARE_ACCURACY:
            if n1.x < n2.x:
                return True
            else:
                return False
        else:
            if abs(n1.y - n2.y) > NODE_COMPARE_ACCURACY:
                if n1.y < n2.y:
                    return True
                else:
                    return False
            else:
                if abs(n1.z - n2.z) > NODE_COMPARE_ACCURACY:
                    if n1.z < n2.z:
                        return True
                    else:
                        return False
                else:
                    return False


class Grid:
    __doc__ = "Class describing triangular triangular_grid2"

    def __init__(self):
        """
        Grid constructor.
        """
        self.Faces = list()
        self.Edges = list()
        self.Nodes = list()
        self.Zones = list()
        self.avl = AVLTree()
        self.number_of_border_nodes = 0

    def init_zone(self):
        """
        Init zone 1 of triangular_grid2.

        Makes all elements of the triangular_grid2 belong to zone 1.
        """
        z = Zone()
        z.Nodes = self.Nodes
        z.Faces = self.Faces
        self.Zones.append(z)

    @staticmethod
    def link_face_and_node(f, n):
        """
        Link node and face.
        :param n: node.
        :param f: face.
        """
        assert len(f.nodes) < 3, 'There are already 3 nodes incident to the face'
        n.faces.append(f)
        f.nodes.append(n)

    @staticmethod
    def link_node_and_edge(n, e):
        """
        Link node and edge.
        :param n: node.
        :param e: edge.
        """
        assert len(e.nodes) < 2, 'There are already 2 nodes incident to the edge'
        n.edges.append(e)
        e.nodes.append(n)

    @staticmethod
    def link_face_and_edge(f, e):
        """
        Link face and edge.
        :param f: face.
        :param e: edge.
        """
        assert len(f.edges) < 3, 'There are already 3 edges incident to the face'
        assert len(e.faces) < 2, 'There are already 2 faces incident to the edge'
        if e not in f.edges:
            f.edges.append(e)
        if f not in e.faces:
            e.faces.append(f)

    @staticmethod
    def is_edge_present(n1, n2):
        """
        Whether the `triangular_grid2.Edges` contains the edge connecting nodes n1 and n2.
        :param n1: node 1.
        :param n2: node 2.
        :return: id of the edge if present and None if not.
        """

        for edge in n1.edges:
            if n2 in edge.nodes:
                return edge

        return None

    def set_nodes_and_faces(self, x, y, z, triangles):
        """
        Create Grid's nodes from the x, y, z co-ordinates.
        :param x, y, z: arrays of coordinates
        :param triangles: array (n_tri, 3) with indexes of corresponding nodes.

        Fills all faces' values with face's id.
        """
        j = 0
        for x, y, z in zip(x, y, z):
            self.Nodes.append(Node(x, y, z, j))
            j += 1

        i = 0
        for nids in triangles:
            f = Face(i)
            i += 1
            f.nodes_ids = nids + 1
            self.link_face_and_node(f, self.Nodes[nids[0]])
            self.link_face_and_node(f, self.Nodes[nids[2]])
            self.link_face_and_node(f, self.Nodes[nids[1]])
            self.Faces.append(f)

        self.init_zone()

    def relocate_values_from_faces_to_nodes(self):
        """First use the simplest algrithm.
        The value in the node is a mean of values in the adjacent faces."""
        for n in self.Nodes:
            n_faces = len(n.faces)
            t = 0
            hw = 0
            for f in n.faces:
                t += f.T
                hw += f.Hw
            n.T = t / n_faces
            n.Hw = hw / n_faces

    def relocate_values_from_nodes_to_faces(self):
        """Set values in faces as mean of the neighbour nodes."""
        for f in self.Faces:
            f.T = (f.nodes[0].T + f.nodes[1].T + f.nodes[2].T) / 3.0
            f.Hw = (f.nodes[0].Hw + f.nodes[1].Hw + f.nodes[2].Hw) / 3.0

    def values_from_nodes_to_array(self) -> array:
        """Return nodes' values as an array."""
        res = list()
        for n in self.Nodes:
            res.append(n.value)

        return array(res)

    def return_coordinates_as_a_ndim_array(self) -> array:
        """Return (n_points, 3) array of coordinates of nodes."""
        x, y, z = list(), list(), list()

        for n in self.Nodes:
            x.append(n.x)
            y.append(n.y)
            z.append(n.z)

        return array([x, y, z]).T

    def return_aux_nodes_as_a_ndim_array(self) -> array:
        """Return (n_points, 3) array of coordinates of nodes."""
        x, y, z = list(), list(), list()

        for f in self.Faces:
            x.append(f.aux_node.x)
            y.append(f.aux_node.y)
            z.append(f.aux_node.z)

        return array([x, y, z]).T

    def make_avl(self):
        """Compose an avl tree that contains references to nodes and allows to logn search."""
        for n in self.Nodes:
            self.avl.insert(n)

    def is_isomprphic_to(self, grid) -> bool:
        """
        Check whether the triangular_grid2 is isomorphic to another triangular_grid2.

        We make a simple two-steps check:
        1. Whether the number of nodes is equal.
        2. Whether the sorted arrays of nodes' degrees (number of connected nodes) of the grids are equal.

        It means that the result is not hundred percent correct, but this heuristics are useful and easy.

        :param grid: Grid (obj)
        :return: bool
        """
        if len(self.Nodes) == len(grid.Nodes):
            degrees_old = []
            degrees_new = []

            for n in self.Nodes:
                degrees_old.append(len(n.edges))
            for n in grid.Nodes:
                degrees_new.append(len(n.edges))

            sorted(degrees_new)
            sorted(degrees_old)

            if degrees_new == degrees_old:
                return True

        return False

    def relocate_values_from_isomorphic_grid(self, grid):
        for i in range(len(self.Faces)):
            self.Faces[i].T = grid.Faces[i].T
            self.Faces[i].Hw = grid.Faces[i].Hw

    def compute_aux_nodes(self):
        """Calculate the points which are the point of medians' intersection."""
        for f in self.Faces:
            n1, n2, n3 = f.nodes[0], f.nodes[1], f.nodes[2]
            f.aux_node.x = (n1.x + n2.x + n3.x) / 3
            f.aux_node.y = (n1.y + n2.y + n3.y) / 3
            f.aux_node.z = (n1.z + n2.z + n3.z) / 3

    def return_paramenter_as_ndarray(self, parameter):
        values_in_auxes = list()
        for f in self.Faces:
            if parameter == 'T':
                values_in_auxes.append(f.T)
            if parameter == 'Hw':
                values_in_auxes.append(f.Hw)
        return array(values_in_auxes).reshape((len(values_in_auxes), 1))

    def set_aux_nodes_parameters(self, interpolated_parameters, parameter='T'):
        assert interpolated_parameters.shape[0] == 1, 'Wrong array dimensions'
        for i, f in enumerate(self.Faces):
            if parameter == 'T':
                f.T = interpolated_parameters[0, i]
            if parameter == 'Hw':
                f.Hw = interpolated_parameters[0, i]

    def depth_first_traversal(self, node, component):
        if node.component is None:
            node.component = component
        else:
            return

        neighbours = []
        for e in node.edges:
            assert len(e.nodes) == 2
            if e.nodes[0] == node:
                neighbours.append(e.nodes[1])
            else:
                neighbours.append(e.nodes[0])

        for neigh in neighbours:
            self.depth_first_traversal(neigh, component)

    def get_number_of_components(self):
        assert len(self.Nodes) > 0
        for i in range(1, 1000):
            start_node = None
            for n in self.Nodes:
                if n.component is None:
                    start_node = n
                    break

            if start_node is None:
                return i - 1
            self.depth_first_traversal(start_node, i)

    def mean_alpha_quality_measure(self):
        assert len(self.Faces) > 0
        mean_alpha_quality_measure = 1
        min_alpha_quality_measure = inf
        for f in self.Faces:
            aqm = f.alpha_quality_measure()
            mean_alpha_quality_measure *= aqm
            if aqm < min_alpha_quality_measure:
                min_alpha_quality_measure = aqm
        print('mean alpha: {}\nmin alpha: {}'.format(mean_alpha_quality_measure ** (1 / len(self.Faces)),
                                                     min_alpha_quality_measure))


class Zone:
    __doc__ = 'Class describing triangular_grid2 zone'

    def __init__(self):
        self.Nodes = None
        self.Faces = None
