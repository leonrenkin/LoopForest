import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Iterable, Callable, Union, Sequence, Set
from collections import defaultdict
from numpy.typing import NDArray
import itertools
import numpy as np
import gudhi as gd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.collections import LineCollection, PolyCollection
import time
import seaborn as sns
import bisect
from bisect import bisect_right

# ------- helper function -----------

def key(simplex):
    """ canonical key: return sorted tuple of vertex ids so orientation doesn’t matter """ 
    return tuple(sorted(simplex))

def sign_of_determinant(vectors):
    """
    Computes the sign of the determinant of d vectors in R^d.
    Returns:
        +1 if det > 0
        -1 if det < 0
        0  if det = 0
    """
    A = np.array(vectors, dtype=float)
    d = A.shape[0]

    sign = 1

    for i in range(d):
        # Find pivot
        pivot = i + np.argmax(abs(A[i:, i]))
        if abs(A[pivot, i]) < 1e-12:
            return 0  # determinant is zero

        # Row swap changes sign
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            sign *= -1

        # Eliminate below pivot
        for j in range(i + 1, d):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]

    # Sign of determinant is the product of the signs of diagonal entries
    diag_sign = np.sign(np.prod(np.sign(np.diag(A))))
    return int(sign * diag_sign)

def are_dict_keys_sorted(d):
    """Return True if dict keys are in ascending order (linear time)."""
    it = iter(d)  # iterates over keys in insertion order
    try:
        prev = next(it)
    except StopIteration:
        return True  # empty dict is trivially sorted

    for k in it:
        if k < prev:
            return False
        prev = k
    return True

def simplex_orientation(simplex, point_cloud):
    vectors = [point_cloud[i]-point_cloud[simplex[0]] for i in simplex[1:]]
    return sign_of_determinant(vectors=vectors)

def signed_boundary(simplex: List[int], orientation: int): 
    return {(tuple(simplex[:i] + simplex[i+1:]), orientation* (-1)**i) for i in range(len(simplex))}

# ------------ Classes ---------------

@dataclass
class SignedChain: 
    signed_simplices: Set[tuple]        # oriented d-simplices, stored as (simplex, orientation)
    active_start: float = float("-inf")
    active_end: float   = float("-inf")

    def merge_at_simplex(self, cycle: "SignedChain", simplex: list[int]) -> "SignedChain":
        union = self.signed_simplices | cycle.signed_simplices
        return SignedChain(signed_simplices= union.difference({(tuple(simplex),1),(tuple(simplex),-1)}) )
    
    def cancel_simplex(self, simplex: list[int]) -> "SignedChain":
        return SignedChain(signed_simplices= self.signed_simplices.difference({(tuple(simplex),1),(tuple(simplex),-1)}) )
    
    def without_double_edges(self) -> "SignedChain":
        """
        Return a new SignedChain where edges that appear with opposite
        orientations cancel out.

        If an underlying simplex appears only with one orientation, it is kept.
        """
        # aggregate orientations per underlying simplex
        coeffs: Dict[tuple, int] = {}
        for simplex, orientation in self.signed_simplices:
            coeffs[simplex] = coeffs.get(simplex, 0) + int(orientation)

        cleaned: Set[tuple] = set()
        for simplex, c in coeffs.items():
            if c > 0:
                cleaned.add((simplex, 1))
            elif c < 0:
                cleaned.add((simplex, -1))
            # if c == 0, the +1 and -1 cancel -> drop this edge completely
        
        return SignedChain(
            signed_simplices=cleaned,
            active_start=self.active_start,
            active_end=self.active_end,
        )

    def segments(self, point_cloud: NDArray):
        segments = []
        for signed_simplex in self.signed_simplices:
            if signed_simplex[1]==1:
                segments.append(np.array(point_cloud[list(signed_simplex[0])]))
            else:
                segments.append(np.array(point_cloud[list(reversed(signed_simplex[0]))]))
        return segments
    
    def dim(self):
        for signed_simplex in self.signed_simplices:
            return len(signed_simplex[0])-1

    def polyhedral_paths(self, point_cloud: NDArray) -> List[NDArray[np.int32]]:
        """
        Decompose a 1-dimensional SignedChain (collection of oriented edges in R^2)
        into polyhedral paths, choosing at each branching point the leftmost
        outgoing edge (smallest counterclockwise angle).

        Parameters
        ----------
        point_cloud : ndarray, shape (n_points, dim>=2)
            Ambient point cloud. Vertex indices in the simplices refer into this
            array. Only the first two coordinates (x,y) are used.

        Returns
        -------
        paths : list of 1D int ndarrays
            Each array `v = paths[k]` is a cyclic list of vertex indices,
            analogous to `Loop.vertex_list` in LoopForest:
                edges are (v[i], v[i+1]) and the final edge (v[-1], v[0]).
            The first vertex is repeated at the end if the greedy walk closes
            up naturally.
        """
        if not self.signed_simplices:
            return []

        if self.dim() !=1:
            raise ValueError(f"Polyhedral path methods only works for Signed 1-chains. Dimemsion of chain: {self.dim}")

        from collections import defaultdict

        def _start_end(simplex: Tuple[int, ...], orientation: int) -> Tuple[int, int]:
            """Return (start, end) vertex of an oriented 1-simplex."""
            if len(simplex) != 2:
                raise ValueError(
                    "polyhedral_paths is only implemented for 1-chains (edges)."
                )
            a, b = simplex
            if orientation == 1:
                return a, b
            elif orientation == -1:
                return b, a
            else:
                raise ValueError("Orientation must be ±1.")

        def _angle_ccw(prev_vec: np.ndarray, next_vec: np.ndarray) -> float:
            """
            Signed angle from prev_vec to next_vec in [0, 2π),
            measured counterclockwise.
            """
            # Work in 2D; take first two coordinates
            pv = prev_vec[:2]
            nv = next_vec[:2]

            # Skip zero-length directions at caller level
            cross = pv[0] * nv[1] - pv[1] * nv[0]
            dot   = pv[0] * nv[0] + pv[1] * nv[1]
            angle = math.atan2(cross, dot)  # ∈ (-π, π]
            if np.all(pv == -nv):
                angle = -math.pi
            return angle

        # Precompute adjacency: start vertex -> list of (signed_simplex, end_vertex)
        edges_by_start: Dict[int, List[Tuple[tuple, int]]] = defaultdict(list)
        for signed_simplex in self.signed_simplices:
            simplex, orientation = signed_simplex
            start, end = _start_end(simplex, orientation)
            edges_by_start[start].append((signed_simplex, end))

        visited: Set[tuple] = set()
        paths: List[NDArray[np.int32]] = []

        for signed_simplex in self.signed_simplices:
            if signed_simplex in visited:
                continue

            simplex, orientation = signed_simplex
            start, end = _start_end(simplex, orientation)

            # Start a new path with this edge
            path_vertices: List[int] = [int(start), int(end)]
            visited.add(signed_simplex)

            prev_vertex = start
            cur_vertex = end

            p_prev = np.asarray(point_cloud[prev_vertex], dtype=float)[:2]
            p_cur  = np.asarray(point_cloud[cur_vertex],  dtype=float)[:2]
            prev_vec = p_cur - p_prev

            while True:
                candidates = edges_by_start.get(cur_vertex, [])
                if not candidates:
                    # No outgoing edges from this vertex
                    raise ValueError("No outgoing edges, this should not happen")

                best_edge: Optional[tuple] = None
                best_next_vertex: Optional[int] = None
                best_angle: Optional[float] = None

                for edge_key, next_vertex in candidates:
                    p_next = np.asarray(point_cloud[next_vertex], dtype=float)[:2]
                    next_vec = p_next - p_cur

                    # Ignore degenerate directions
                    if np.allclose(next_vec, 0.0):
                        continue

                    angle = _angle_ccw(prev_vec, next_vec)

                    if best_angle is None or angle > best_angle:
                        best_angle = angle
                        best_edge = edge_key
                        best_next_vertex = int(next_vertex)

                if best_edge is None or best_next_vertex is None:
                    # Only degenerate candidates
                    raise ValueError("Only degenerate candidates, this should not happen")

                # Stop if we would traverse an already covered signed simplex
                if best_edge in visited:
                    break

                # Advance along the chosen leftmost edge
                path_vertices.append(best_next_vertex)
                visited.add(best_edge)

                prev_vertex, cur_vertex = cur_vertex, best_next_vertex
                p_prev, p_cur = p_cur, np.asarray(point_cloud[cur_vertex], dtype=float)[:2]
                prev_vec = p_cur - p_prev

            paths.append(np.array(path_vertices, dtype=np.int32))

        return paths



@dataclass
class PFNode:
    """ Objects which are the nodes in the LoopForest graph. 
    Each node has a loop representative."""
    id: int #does not need to know its own node
    filt_val: float
    cycle: SignedChain                                                  #Loops are saved as list of indices of simplex
    children: set[int]                                    #ids of children
    parent: Optional[int] = None
    #is_root: bool = True  #True if it is the root of a tree, also used for bookkeeping of active loops
    _barcode_covered: bool = False

    def __repr__(self) -> str:
        return f"Node(id={self.id}, f={self.filt_val})"

class PFBar:
    """ 
    Object which stores a bar in H1 persistence together with a progression of cycle reps.
    Each cycle rep has a an active_start and active_end attribute which is the interval in which this representative is optimal.
    The cycle reps are a strictly decreasing chain w.r.t. inclusion.
    """

    def __init__(self, birth: float, 
                 death: float, 
                 _node_progression: tuple[int,...], 
                 cycle_reps: list[SignedChain], 
                 is_max_tree_bar: Optional[bool]=None, 
                 root_id: Optional[int]=None):
        self.birth = birth
        self.death = death
        self._node_progression = _node_progression #nodes saved as node_ids
        self.cycle_reps = cycle_reps
        self.is_max_tree_bar = is_max_tree_bar
        self.root_id = root_id

    def cycle_at_filtration_value(self, filt_val)->SignedChain:
        """Binary search to find active loop at filtration value of this bar."""

        if filt_val < self.birth:
            raise ValueError(f"Filtration value {filt_val} is too small and not in lifespan of the bar")
        if filt_val >= self.death:
            raise ValueError(f"Filtration value {filt_val} is too large and not in lifespan of the bar")

        if len(self._node_progression)==1:
            return self.cycle_reps[0]

        first = 0
        last = len(self.cycle_reps)-1
        best = None

        while first<=last:
            midpoint = (first+last) // 2

            if self.cycle_reps[midpoint].active_start <= filt_val:
                best = self.cycle_reps[midpoint]
                first = midpoint+1
            else:
                last = midpoint-1

        if best == None:
            raise ValueError(f"Binary search in barcode returned {None}, check if bar is empty list")
        elif best.active_start>filt_val or best.active_end<= filt_val:
            raise ValueError("Output of binary search incorrect, loop not active at filtration value. Check correctness of loop_at_filtration method.")
        
        return best
    
    def lifespan(self):
        """Returns lifespan of bar"""
        return self.death - self.birth
    
@dataclass
class StepFunctionData:
    """
    Representation of a piecewise-constant function:

    f(t) = vals[i]  on [starts[i], ends[i]]
         = baseline elsewhere.

    All arrays are 1D and aligned by index. `domain` is the global range
    where the function is potentially non-zero (for convenience).
    """
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    vals: NDArray[np.float64]
    baseline: float
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

@dataclass
class PiecewiseLinearFunction:
    """
    Piecewise-linear function specified by breakpoints (xs, ys).
    Between xs[i] and xs[i+1] the function is linear. Outside [xs[0], xs[-1]]
    the function evaluates to 0.0 by default.
    """
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __call__(self, x: Union[NDArray[np.float64], float]) -> Union[NDArray[np.float64], float]:
        if self.xs.size == 0:
            if np.isscalar(x):
                return 0.0
            x_arr = np.asarray(x, dtype=float)
            return np.zeros_like(x_arr)

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.interp(x_arr, self.xs, self.ys, left=0.0, right=0.0)

        if np.isscalar(x):
            # np.interp returns a scalar np.ndarray for scalar input
            return float(y_arr)
        return y_arr

class PersistenceForest:
    """Object that computes and stores progression of optimal cycles for alpha complex of a point cloud in a forest format."""

    def __init__(self, 
                 point_cloud,
                 compute = True,
                 reduce: bool = True,
                 compute_barcode: bool = True,
                 print_info: bool = False) -> None:
        self.point_cloud = np.array(point_cloud) #point cloud is list of n-dim arrays

        #check if point cloud has correct shape
        self.dim = self.point_cloud.shape[1]
        if print_info:
            print(f'dimension = {self.dim}')

        self._node_id = itertools.count(1)         #used to assign unique id to each node in the forest
        self._cycle_id = itertools.count(1)
        self.nodes: Dict[int, PFNode] = {}
        self.cycles: Dict[int, SignedChain] = {}

        self._active_node_ids: set[int] = set()               #used for bookkeeping of active nodes in forest computation algorithm
        self.roots: set[int]  = set()                 #List of roots of trees in the forest

        self.levels: list[float] = []               #Critical filtration values, might give duplicates in current implementation

        start = time.perf_counter()
        self._alpha_complex = gd.AlphaComplex(points=point_cloud) # pyright: ignore[reportAttributeAccessIssue]
        self.simplex_tree = self._alpha_complex.create_simplex_tree()
        alpha_complex_time = time.perf_counter()-start
        if print_info:
            print(f"Alpha complex generated in {alpha_complex_time}")

        start = time.perf_counter()
        #take square root of filtration value since filtation values for alpha complexes are squared in Gudhi
        for simplex, filtration in self.simplex_tree.get_filtration():
            self.simplex_tree.assign_filtration(simplex, (filtration**0.5)*2)
        
        # Extract s filtration up to order d
        self.filtration =  [(simplex,filtration) for simplex, filtration in self.simplex_tree.get_filtration() if len(simplex) >= self.dim] #keep simplices up to codim 1
        filtration_time = time.perf_counter()-start
        if print_info:
            print(f"Filtration processed in {filtration_time}")

        self.barcode: set[PFBar] = set()

        #compute forest
        if compute:
            self._compute_forest(reduce=reduce, compute_barcode=compute_barcode, print_info = print_info)
            self.reduced = reduce
        

    # ---------- builders ---------


    def add_leaf(self, simplex: List[int], filt_val: float, orientation: int):
        """
        Create a new leaf in tree. 
        Corresponds to death of a cycle in homology and adding a loop in algorithm for computing the LoopForest.
        """

        nid = next(self._node_id)

        new_cycle = SignedChain(signed_simplices=signed_boundary(simplex=simplex,orientation=orientation))

        new_node = PFNode(id=nid,filt_val=filt_val, children=set(), cycle=new_cycle)

        self.nodes[nid]=new_node
        self._active_node_ids.add(nid)  #roots are active nodes in algorithm, at termination the active nodes are precicely the roots of the forest 

        self.levels.append(filt_val)

        return new_node
    
    def make_root(self, node: PFNode, filt_val: float):
        """
        Ends this tree in the forest by creating the root as a top node.
        Corresponds to birth in homology and removing an edge of a loop without merging it with another loop in LoopForest computation algorithm.
        """
        nid = next(self._node_id)

        self._active_node_ids.remove(node.id)
        node.parent = nid #new root node is parent of input node

        root_node = PFNode(id=nid, filt_val=filt_val, cycle=node.cycle, children={node.id})


        self.nodes[nid]=root_node
        self.roots.add(root_node.id)

        self.levels.append(filt_val)

        return

    def merge_nodes(self, node1: PFNode, node2: PFNode, parent_cycle: SignedChain, filt_val: float):
        """ 
        Creates parent node of node1 and node2 with loop representative parent loop
        Corresponds to a loop being split into two loops in homology and a new bar appearing in barcode
        """

        nid = next(self._node_id)
        node1.parent = nid
        node2.parent = nid

        parent_node = PFNode(id=nid, filt_val=filt_val, children={node1.id,node2.id}, cycle=parent_cycle)
        self.nodes[nid]=parent_node
        
        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node1.id)
        self._active_node_ids.remove(node2.id)

        self.levels.append(filt_val)

        return
    
    def update_node(self, node: PFNode, updated_cycle: SignedChain, filt_val:float):
        """ Updates loop representative, corresponds to node with one parent and one child in tree """

        nid = next(self._node_id)
        node.parent=nid

        update_node = PFNode(id=nid, filt_val=filt_val, children={node.id},cycle = updated_cycle)
        self.nodes[nid]=update_node

        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node.id)

        self.levels.append(filt_val)

        return
    
    
    # ----- compute the forest ----------

    def _compute_forest(self, reduce = True, compute_barcode = True, print_info: bool = False):
        """ 
        Computes LoopForest object for a point cloud.
        reduce = True means that multiple changes at the same filtration value is collapsed to a single node.
        compute_barcode = True computes barcodes and stores it in self.barcode as list of bar objects
        """

        loop_forest_start = time.perf_counter()

        face_cycle_dict = {}

        #simplices is already ordered in ascending order by number of simplices 
        for simplex, filt_val in reversed(self.filtration):
            
            if len(simplex) == self.dim+1:
                orientation = simplex_orientation(simplex=simplex, point_cloud=self.point_cloud)

                new_node = self.add_leaf( simplex=simplex, filt_val=filt_val, orientation = orientation)

                faces = list(itertools.combinations(simplex, len(simplex)-1 ) )
                for face in faces:
                    if key(face) in face_cycle_dict:
                       face_cycle_dict[key(face)].append(new_node.id)
                    else:
                        face_cycle_dict[key(face)] = [new_node.id]

            elif len(simplex) == self.dim:
                #L is nodes containing simplex, can be of the form [],[l1], [l1,l2], [l1,l1]
                #L is active nodes over L_tmp

                #If key exists, get its value and remove it
                #if key does not exists, get []
                L_tmp_ids = face_cycle_dict.pop(key(simplex), [])

                L = self._update_node_list(L_tmp_ids)


                #if no cycle contains simplex, nothing happens
                if len(L) == 0:
                    continue

                #if cycle is only contained in a single loop and appears only once in that loop once, remove that loop from the active loops 
                elif len(L) == 1:
                    #update the loop dict for all faces which very contained in the loop we just removed
                    for signed_simplex in L[0].cycle.signed_simplices:

                            simplex = signed_simplex[0]

                            L_simplex_tmp = face_cycle_dict.pop(key(simplex), None)
                            if L_simplex_tmp is None:
                                continue

                            L_simplex = self._update_node_list(L_simplex_tmp)

                            if len(L_simplex)> 2:
                                raise ValueError("L_edge too long in loop removal process")

                            if len(L_simplex)==1:
                                continue
                            elif L_simplex[0] != L[0]:
                                face_cycle_dict[key(simplex)] = [L_simplex[0].id]
                            elif L_simplex[1] != L[0]: 
                                face_cycle_dict[key(simplex)] = [L_simplex[1].id]
                            else:
                                continue
                                
                    self.make_root(node=L[0],filt_val=filt_val)

                    continue

                elif len(L) == 2 and L[0]!=L[1]:
                    parent_cycle = L[0].cycle.merge_at_simplex(cycle=L[1].cycle,  simplex=simplex) 
                    self.merge_nodes( node1=L[0], node2=L[1], parent_cycle=parent_cycle, filt_val=filt_val)
                    """if not loop_in_filtration_check(parent_loop.vertex_list, simplex_tree=self.simplex_tree, filt_value=filt_val):
                            print('edge', simplex)
                            print('edge dict entry')
                            print('filtration value', filt_val)
                            print(f'first loop', L[0].cycle)
                            print(f'second loop', L[1].cycle)
                            raise ValueError("Loop not in simplex, Loop concat Case")"""


                elif len(L) == 2 and L[0]==L[1]:
                    #Same simplex is contained in a cycle in both orientations -> we remove it from the the cycle
                    updated_cycle = L[0].cycle.cancel_simplex(simplex=simplex)
                    """if not loop_in_filtration_check(vertex_loop, simplex_tree=self.simplex_tree, filt_value=filt_val):
                                print('edge', simplex)
                                print(f'starting loop', L[0].loop)
                                print(f'outer loop', vertex_loop)
                                raise ValueError("Loop not in simplex, Tiebreak Case")"""
                    

                    self.update_node(node=L[0], updated_cycle=updated_cycle, filt_val=filt_val)

                else:
                    print(L)
                    print(simplex)
                    raise ValueError("Error, L is of the wrong form")

        loop_forest_time = time.perf_counter() - loop_forest_start
        if print_info:
            print(f"Forest succesfully computed in {loop_forest_time} sec")

        #compute where each loop is active
        self._compute_loop_activity()

        if reduce:
            self._reduce_forest(print_info = print_info)

        if compute_barcode:
            self.compute_barcode(print_info = print_info)
        
        return


    # ----- methods to work with the forest

    def active_nodes_at(self, filt_val: float) -> List[PFNode]:
        """
        Return list of active nodes at a given filtration value.

        A node is active at r if:
        - node.filt_val >= r
        - has a parent and parent has filt val < r
        """
        nodes = self.nodes

        active: List[PFNode] = []
        for n in nodes.values():
            if n.filt_val < filt_val:
                continue
            if n.parent == None:
            # all children must exist and be strictly above alpha
                continue
            else:
                parent = nodes[n.parent]
                if parent.filt_val >= filt_val:
                    continue

            active.append(n)
        
        # deterministic order: higher filt_val first, then id
        active.sort(key=lambda n: (-n.filt_val, n.id))
        return active

    def active_cycles_at(self, filt_val: float) -> List[SignedChain]:
        active_nodes = self.active_nodes_at(filt_val=filt_val)
        return [node.cycle for node in active_nodes]

    def leaves_below_node(self, node: PFNode) -> set[int]:
        """ Returns set of all leaves below a given node (below in tree means higher filtration value) """
        leaf_ids: set[int] = set()

        if len(node.children) == 0:
            leaf_ids.add(node.id)
            return leaf_ids
        
        for cid in node.children:
            child = self.nodes[cid]
            leaf_ids.update(self.leaves_below_node(child))


        return leaf_ids

    def leaf_to_node_path(self, leaf: PFNode, node: PFNode) -> List[int]:
        """ 
        Returns direct path from a leaf to a node.
        Path is returned as list of node ids

        If there does not exist a path, an error is raised.
        """
        path = [leaf.id]

        active_node = leaf
        while active_node.parent != None:
            active_node = self.nodes[active_node.parent]
            path.append(active_node.id)
            if active_node.id == node.id:
                return path

        if active_node.id != node.id:
            raise ValueError(f"Node {node} is not above leaf {leaf}")

        return path

    def node_to_leaf_path(self, leaf: PFNode, node: PFNode) -> List[int]:
        """ 
        Returns direct path from a node to a leaf.
        Path is returned as list of node ids

        If there does not exist a path, an error is raised.
        """
        return list(reversed(self.leaf_to_node_path(leaf=leaf, node=node)))

    def get_root(self, node: PFNode) -> PFNode:
        while node.parent != None:
                    pid = node.parent
                    node = self.nodes[pid]

        return node

    def _update_node_list(self, node_id_list: List[int]) -> List[PFNode]:
        """ Returns list of all current roots of a given list of node IDs"""
        L = [ self.get_root( self.nodes[id] ) for id in node_id_list ]
        return L

    # ----- reduce forest (collapses trivial edges which happen at the same filtration value) -------------

    def _collapse_parent_child(self, parent: PFNode, child: PFNode):
        """
        Collapses a parent - child pair into the parent node.
        Intended for parent - child pair with same filtration value 
        """

        parent.children.remove(child.id)
        parent.children.update(child.children)

        #re-parent grandchildren
        for gcid in child.children:
            self.nodes[gcid].parent = parent.id

        #remove child node from forest
        del self.nodes[child.id]

        #if parent is now isolated point in forest, delete it from forest completely
        if len(parent.children)==0 and parent.parent == None:
            del self.nodes[parent.id]
            self.roots.remove(parent.id)

        return

    def _reduce_forest(self, print_info: bool = False):
        """
        Reduce the forest by collapsing every parent–child pair with equal filtration value.

        Collapse rule for an edge (parent p, child c) with p.filt_val == c.filt_val:
        - Keep the *parent* node p (its loop stays as-is).
        - Remove the child node c from the forest.
        - The parent of the resulting (collapsed) node remains p.parent (i.e., the
            parent of the original parent), if any.
        - Children of the resulting node are the union of p.children and c.children,
            minus the removed child c itself.
        - For every grandchild g in c.children, set g.parent = p.id.
        Repeats until no collapsible edges remain.
        """
        if print_info:
            print("Reducing the forest")
        reduction_start = time.perf_counter()

        collapses = 0

        if print_info:
            print(f"Number of nodes before reduction: {len(self.nodes.keys())}")

        #iterate over snapshot of the nodes in the tree, the nodes dict might be changed each iteration
        node_list = list(self.nodes.values())

        for p in node_list:

            # p might have been deleted as a child in a previous iteration of this outer loop
            if p.id not in self.nodes.keys():
                print("continue case")
                continue

            for cid in p.children.copy():
                child = self.nodes[cid]
                if child.filt_val == p.filt_val:
                    self._collapse_parent_child(parent=p, child=child)
                    collapses += 1

        reduction_time = time.perf_counter() - reduction_start
        if print_info:
            print(f"Reduction complete in {reduction_time} sec")
            print(f"Number of nodes after reduction: {len(self.nodes.keys())}")
        
        return

    # If we have multiple edges appearing at the same filtration value, we might get a root node which is also a merge in the reduction process
    # This will lead to a node which appears in the root list but has type merge
    # Not a mistake in the code, simply the way the edge case is currently handled
    # -> use node.parent == None to check if a node is a root
    # If a merge is also a root, then the merge should be split into 2 seperate roots as the merge only lives for 0 time
    # This is not implemented yet and should not occur for points in general position

    # ------ Add active period of each loops ----------

    def _compute_loop_activity(self):
        """ Computes period in which each loop is an optimal cycle rep and adds it to the loop as attributes """

        for node in self.nodes.values():

            if node.parent == None:
                continue
            
            parent = self.nodes[node.parent]
            node.cycle.active_end = node.filt_val
            node.cycle.active_start = parent.filt_val


        return

    # ----- Compute barcode sequence ---------

    def compute_barcode(self, print_info: bool = False):
        """ Computes H1 barcode of forest and stores it in self.barcode """
        
        if print_info:
            print("Computing Barcode")
        barcode_start = time.perf_counter()

        #dict should be ordered with filtration values decreasing since nodes are added in that order
        if not are_dict_keys_sorted(self.nodes):
            raise ValueError("Node dict keys are not sorted. This should not happen. Easy fix: sort keys in compute_barcode function (currently not implemented)")
    

        for id, node in self.nodes.items():
            #every barcode starts at leaf
            if len(node.children)>0:
                continue

            death = node.filt_val
            node_id_progession = [id]
            cycle_progression = [node.cycle]
            node._barcode_covered=True
            is_max_tree_bar = True
            root_id = self.get_root(node).id

            if node.parent == None:
                raise ValueError("Leaf has no Parent, this should not happen")
            else:
                parent = self.nodes[node.parent]

            #walk up forest until a root or an already _barcode_covered node is discovered
            while parent.parent is not None:
                #check if parent node has already been covered by leaf with larger filtration value
                if parent._barcode_covered == True:
                    is_max_tree_bar = False
                    break

                node_id_progession.append(parent.id)
                cycle_progression.append(parent.cycle)
                parent._barcode_covered = True

                #move to parent of parent
                parent = self.nodes[parent.parent]

            birth = parent.filt_val


            #reverse lists to get progression which is ascending with respect to filtration value
            bar = PFBar(birth=birth,
                      death=death, 
                      _node_progression = tuple(reversed(node_id_progession)), 
                      cycle_reps=list(reversed(cycle_progression)), 
                      is_max_tree_bar=is_max_tree_bar,
                      root_id=root_id)
            self.barcode.add(bar)
 

        barcode_time = time.perf_counter() - barcode_start
        if print_info:
            print(f"Barcode computation completed in {barcode_time} sec")
    
        return
         
    def max_bar(self):
        return max(self.barcode, key=lambda bar: bar.lifespan())
    
    def active_bars_at(self, filt_val:float):
        return [bar for bar in self.barcode if (bar.birth<=filt_val and bar.death>filt_val)]

    # ------ generate color scheme  ---------

    def _build_color_map_forest(self, seed: Optional[int] = 39, start_color: Optional[str] = "#ff7f0e",):
        """
        Computes a color map which assign a color to each bar in the barcode. 
        Bars in same tree will have similiar colors. 
        Saved as a dictionary {bar: "#RRGGBB"} in self.color_map_forest
        Based on json
        Seed for randomness
        """

        from color_scheme import color_map_for_bars

        ordered_bars = sorted(list(self.barcode), key= lambda bar: bar.lifespan(), reverse=True)

        self.color_map_forest = color_map_for_bars(
            ordered_bars,
            seed =seed,
            by_id=False,
            prefer_start=start_color
        )

        return

    def _build_color_map_bars(self):
        """
        Computes a color map which assign a color to each bar in the barcode. 
        Ignores tree structure and cycles through 20 colors from longest to shortest bar 
        Saved as a dictionary {bar: "#RRGGBB"} in self.color_map_bars
        """
        bars_sorted = sorted(list(self.barcode), key = lambda bar:bar.lifespan(), reverse=True )
        colors = sns.color_palette("tab20", len(bars_sorted))

        self.color_map_bars = {bars_sorted[i]: colors[i] for i in range(len(bars_sorted))}
        return

    # ----- plotting tools -------

    def plot_at_filtration(
        self,
        filt_val: float,
        ax=None,
        show: bool = True,
        fill_triangles: bool = True,
        figsize: tuple[float, float] = (7, 7), 
        point_size: float = 3,
        coloring: Literal['forest','bars'] = "forest",
        title: Optional[str] = None,
        loop_edge_arrows: bool = False,
        remove_double_edges: bool = False,
    ):
        """
        Plot the 2-D point cloud, all edges/triangles with filtration <= filt_val,
        and overlay the loops of the nodes active at filt_val.

        Notes
        -----
        - GUDHI's AlphaComplex / SimplexTree work
        Pass the same units here.
        - Uses SimplexTree.get_filtration(), which is sorted by increasing filtration.

        Parameters
        ----------
        filt_val : float
            Filtration threshold.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; if None, a new figure+axes are created.
        show : bool
            If True, calls plt.show() when done.
        fill_triangles : bool
            If True, lightly fill triangles present at this filtration.
        loop_edge_arrows : bool
            If True, draw small arrows along each loop edge to indicate
            the orientation of the cycle representatives.

        Returns
        -------
        matplotlib.axes.Axes
        """
        
        if self.dim != 2:
            raise ValueError("plot_at_filtration only implemented for dimension 2")
        
        if coloring == "forest":
            #built color map if it has not already been done
            if not hasattr(self,"color_map_forest"):
                self._build_color_map_forest()
            color_map = self.color_map_forest
        elif coloring == "bars":
            if not hasattr(self,"color_map_bars"):
                self._build_color_map_bars()
            color_map = self.color_map_bars

        # --- Prep
        pts = np.asarray(self.point_cloud, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("point_cloud must be an (n_points, 2) array-like.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # --- Collect edges and triangles present at this filtration value
        edges_xy = []      # list of [[x1,y1],[x2,y2]]
        tris_xy = []       # list of [[x1,y1],[x2,y2],[x3,y3]]
        for simplex, f in self.filtration:
            if f > filt_val:
                # Filtration is sorted non-decreasing → safe to stop here
                break
            if len(simplex) == 2:  # edge
                i, j = simplex
                edges_xy.append([pts[i], pts[j]])
            elif len(simplex) == 3:  # triangle
                i, j, k = simplex
                tris_xy.append([pts[i], pts[j], pts[k]])

        # --- Base scatter
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, color="k", zorder=3, label="points")

        # --- Draw triangles first (under edges)
        if fill_triangles and tris_xy:
            tri_coll = PolyCollection(
                tris_xy, closed=True, edgecolors="none", facecolors="C0", alpha=0.15, zorder=1
            )
            ax.add_collection(tri_coll)

        # --- Draw edges
        if edges_xy:
            edge_coll = LineCollection(edges_xy, linewidths=0.8, colors="0.3", zorder=2, label="edges")
            ax.add_collection(edge_coll)

        

        # --- Overlay loops from active nodes at filt_val
        for bar in self.barcode:
            if filt_val>=bar.birth and filt_val<bar.death:

                cycle = bar.cycle_at_filtration_value(filt_val=filt_val)    

                # >>> make a Sequence[ArrayLike] (list of 2x2 arrays) for Pylance
                if remove_double_edges:
                    segments = cycle.without_double_edges().segments(point_cloud=self.point_cloud)
                else:
                    segments = cycle.segments(point_cloud=self.point_cloud)

                # Thicker colored edges along the loop
                loop_coll = LineCollection(segments, linewidths=1.8, colors=[color_map[bar]], zorder=5)
                ax.add_collection(loop_coll)

                # Optional arrows to show loop edge orientation
                if loop_edge_arrows:
                    for seg in segments:
                        # seg is a 2x2 array: [start, end]
                        (x0, y0), (x1, y1) = np.asarray(seg, dtype=float)

                        dx = x1 - x0
                        dy = y1 - y0
                        length = float(np.hypot(dx, dy))
                        if length == 0.0:
                            continue  # skip degenerate segments

                        # Place arrow around the middle of the segment, slightly shortened
                        frac = 0.5  # fraction of the segment length used for the arrow body
                        mx = 0.5 * (x0 + x1)
                        my = 0.5 * (y0 + y1)

                        # Direction unit vector
                        ux = dx / length
                        uy = dy / length

                        half = 0.5 * frac * length
                        x_start = mx - ux * half
                        y_start = my - uy * half
                        x_end   = mx + ux * half
                        y_end   = my + uy * half

                        ax.annotate(
                            "",
                            xy=(x_end, y_end),
                            xytext=(x_start, y_start),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                linewidth=1.6,
                                color=color_map[bar],
                                mutation_scale=10
                            ),
                            zorder=6,
                    )

        # --- Aesthetics
        ax.set_aspect("equal", adjustable="box")
        if title is None:
            ax.set_title(f"α ≤ {filt_val:.4g}  •  edges/triangles in filtration + active loops")
        else:
            ax.set_title(title)
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        # A simple legend (points + edges); loop colors are self-explanatory on top
        #handles, labels = ax.get_legend_handles_labels()
        #if handles:
        #    ax.legend(loc="lower right", frameon=True)

        ax.autoscale()  # fit collections
        if show:
            plt.show()
        return ax

    def plot_at_filtration_with_dual( 
        self,
        filt_val: float,
        ax=None,
        fill_triangles: bool = True,
        figsize: tuple[float, float] = (7, 7),
        point_size: float = 3,
        coloring: Literal['forest','bars'] = "forest",
        dual_vertex_size: float = 26,
    ):
        if coloring == "forest":
            if not hasattr(self, "color_map_forest"):
                self._build_color_map_forest()
            color_map = self.color_map_forest
        elif coloring == "bars":
            if not hasattr(self, "color_map_bars"):
                self._build_color_map_bars()
            color_map = self.color_map_bars
        else:
            raise ValueError("Unsupported coloring option. Use 'forest' or 'bars'.")

        pts = np.asarray(self.point_cloud, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("point_cloud must be an (n_points, 2) array-like.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        edge_info: dict[Tuple[int, int], tuple[np.ndarray, float]] = {}
        triangle_info: dict[Tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
        edge_to_triangles: dict[Tuple[int, int], list[Tuple[int, int, int]]] = defaultdict(list)
        vertex_to_dual_all: dict[int, list[tuple[np.ndarray, float]]] = {i: [] for i in range(len(pts))}

        for simplex, filtration in self.filtration:
            if len(simplex) == 2:
                i, j = simplex
                edge_key = tuple(sorted((i, j)))
                segment = np.array([pts[i], pts[j]])
                edge_info[edge_key] = (segment, filtration)
            elif len(simplex) == 3:
                i, j, k = simplex
                tri_key = tuple(sorted((i, j, k)))
                coords = np.array([pts[i], pts[j], pts[k]])
                barycenter = coords.mean(axis=0)
                triangle_info[tri_key] = (coords, barycenter, filtration)
                for v in tri_key:
                    vertex_to_dual_all[v].append((barycenter, filtration))
                for edge in itertools.combinations(tri_key, 2):
                    edge_key = tuple(sorted(edge))
                    edge_to_triangles[edge_key].append(tri_key)

        edges_present: list[np.ndarray] = []
        edges_future: list[np.ndarray] = []
        for edge_key, (segment, filtration) in edge_info.items():
            if filt_val >= filtration:
                edges_present.append(segment)
            else:
                edges_future.append(segment)

        tris_present: list[np.ndarray] = []
        tris_pending: list[np.ndarray] = []
        dual_vertices_present: list[np.ndarray] = []
        dual_vertices_future: list[np.ndarray] = []
        for coords, barycenter, filtration in triangle_info.values():
            if filtration <= filt_val:
                tris_present.append(coords)
                dual_vertices_future.append(barycenter)
            else:
                tris_pending.append(coords)
                dual_vertices_present.append(barycenter)

        dual_edges_present: list[np.ndarray] = []
        dual_edges_future: list[np.ndarray] = []
        for edge_key, tri_keys in edge_to_triangles.items():
            if len(tri_keys) != 2:
                continue  # boundary edge → no dual edge
            edge_data = edge_info.get(edge_key)
            if edge_data is None:
                continue
            edge_filtration = edge_data[1]
            tri1 = triangle_info.get(tri_keys[0])
            tri2 = triangle_info.get(tri_keys[1])
            if tri1 is None or tri2 is None:
                continue
            bary1 = tri1[1]
            bary2 = tri2[1]
            segment = np.array([bary1, bary2])
            if filt_val < edge_filtration:
                dual_edges_present.append(segment)
            else:
                dual_edges_future.append(segment)

        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, color="k", zorder=3, label="points")

        if fill_triangles and tris_present:
            tri_coll = PolyCollection(
                tris_present,
                closed=True,
                edgecolors="none",
                facecolors="C0",
                alpha=0.28,
                zorder=1,
            )
            ax.add_collection(tri_coll)

        # if dual_vertices_pending:
        #     pending_arr = np.array(dual_vertices_pending)
        #     ax.scatter(
        #         pending_arr[:, 0],
        #         pending_arr[:, 1],
        #         s=dual_vertex_size,
        #         c="C3",
        #         alpha=0.6,
        #         marker="^",
        #         edgecolors="white",
        #         linewidths=0.4,
        #         zorder=3.2,
        #         label="dual vertices (pending)",
        #     )

        if dual_vertices_present:
            present_arr = np.array(dual_vertices_present)
            ax.scatter(
                present_arr[:, 0],
                present_arr[:, 1],
                s=dual_vertex_size * 0.75,
                c="C3",
                marker="o",
                edgecolors="none",
                zorder=2.8,
            )

        if edges_future:
            future_edge_coll = LineCollection(
                edges_future,
                linewidths=0.9,
                colors="0.45",
                alpha=0.5,
                zorder=1.6,
            )
            ax.add_collection(future_edge_coll)

        if edges_present:
            edge_coll = LineCollection(
                edges_present,
                linewidths=0.9,
                colors="0",
                zorder=2,
                label="edges",
            )
            ax.add_collection(edge_coll)

        if dual_edges_future:
            dual_thin_coll = LineCollection(
                dual_edges_future,
                colors="C3",
                linewidths=0.5,
                alpha=0.5,
                linestyle="dotted",
                zorder=3.6,
            )
            ax.add_collection(dual_thin_coll)

        if dual_edges_present:
            dual_thick_coll = LineCollection(
                dual_edges_present,
                colors="C3",
                linewidths=0.5,
                alpha=1,
                zorder=4,
                label="dual edges",
            )
            ax.add_collection(dual_thick_coll)

        for bar in self.barcode:
            if filt_val >= bar.birth and filt_val < bar.death:
                cycle = bar.cycle_at_filtration_value(filt_val=filt_val)
                segments = [np.array(pts[list(signed_simplex[0])]) for signed_simplex in cycle.signed_simplices]    
                loop_coll = LineCollection(segments, linewidths=1.8, colors=[color_map[bar]], zorder=5)
                ax.add_collection(loop_coll)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"α = {filt_val:.4g}")
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(loc="lower right", frameon=True)

        # ax.autoscale()
        return ax


    #ChatGPT plotting function
    def plot_dendrogram(
        self,
        *args,
        **kwargs
    ):
        from forest_plotting import _plot_dendrogram_generic
        return _plot_dendrogram_generic(self, *args, **kwargs)

    def plot_barcode(self, *args, **kwargs):
        """
        Plot a 1D barcode from self.barcode (a set[Bar]).

        Each Bar contributes a horizontal segment from birth to death.
        If death is +inf, an arrow is drawn to the right.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            If given, draw on this axes. Otherwise a new figure/axes is created.
        sort : {"length","birth","death",None}
            Sort bars before plotting (None preserves current order).
            Default is "birth".
        title : str
            Plot title.
        xlabel : str
            Label for the x-axis.
        coloring : {"forest","bars"}
            Which color scheme to use:
            - "forest": use self.color_map_forest (tree-structured colors).
            - "bars":   use self.color_map_bars (ignores tree structure).
            If the chosen color map does not exist yet, it is built as in
            `plot_at_filtration`.
        max_bars : int
            If > 0, display at most this many bars, keeping the longest ones
            (by lifespan). 0 means show all bars.
        min_bar_length : float
            Filter out bars with lifespan < min_bar_length before plotting.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the barcode was drawn on.
        """
        from forest_plotting import _plot_barcode_generic

        return _plot_barcode_generic(self, *args,**kwargs)

    # --------- animation -------------

    def animate_filtration(
        self,
        *args,
        **kwargs
    ):
        """
        Create an animation of the loop forest over the filtration.

        Parameters
        ----------
        filename : str | None, optional
            If given, the animation is written to this path.
        fps : int, optional
            Frames per second for the saved animation.
        frames : int, optional
            Number of time steps (frames) sampled between t_min and t_max.
        with_barcode : bool, optional
            If True, show a second panel with the barcode and a moving vertical
            line indicating the current filtration value.
        t_min, t_max : float | None, optional
            Optional lower/upper bounds on the filtration values to animate.
        dpi : int, optional
            DPI for saving the animation.
        cloud_figsize : (float, float), optional
            Size of the point-cloud panel when with_barcode=False.
        total_figsize : (float, float) | None, optional
            Total figure size when with_barcode=True. If None, a reasonable
            default (10, 5) is used.
        plot_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``plot_at_filtration``.
            For example::
                plot_kwargs=dict(
                    fill_triangles=True,
                    loop_vertex_markers=False,
                    point_size=3,
                    coloring="forest",
                )
        barcode_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``_plot_barcode`` **except**
            ``ax`` and ``coloring``, which are managed by this method.
            For example::
                barcode_kwargs=dict(
                    max_bars=150,
                    min_bar_length=1e-3,
                    sort="length",
                    title="Barcode",
                )

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The created animation. If ``filename`` is not None, the animation
            is also saved to disk.
        fig : matplotlib.figure.Figure
            The figure on which the animation is drawn.
        """
        from forest_plotting import _animate_filtration_generic
        return _animate_filtration_generic(self, *args, **kwargs)

    #------- generalized landscape ----------------

    def plot_barcode_measurement(self, cycle_func, signed = True, bar = None, show =False, *args,**kwargs):
        from forest_landscapes import plot_barcode_measurement_generic

        if signed:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain.without_double_edges(), point_cloud))
            
        return plot_barcode_measurement_generic(forest=self, cycle_func=_cycle_value, bar=bar, show = show,*args,**kwargs)

    def compute_generalized_landscape_family(
        self,
        cycle_func,
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        mode: Literal["raw", "pyramid"] = "pyramid",
        label: Optional[str] = None,
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
        signed: bool = True,
    ):
        """
        Compute a generalized landscape family for this PersistenceForest.

        Parameters
        ----------
        chain_value_func : callable
            A function

                chain_value_func(signed_simplices, point_cloud) -> float

            where `signed_simplices` is typically a list of (simplex, sign)
            pairs from a SignedChain. This lets you define arbitrary functionals
            on chains (e.g. total length, total mass, etc.).
        max_k : int
            Number of landscape levels λ_1..λ_max_k.
        num_grid_points : int, optional
            Number of x-grid samples (if x_grid is None).
        mode : {"raw", "pyramid"}, optional
            Kernel mode; "pyramid" matches the LoopForest convention.
        label : str | None, optional
            Optional label/ID for this forest (stored in the family metadata).
        min_bar_length : float, optional
            Ignore bars with lifespan < min_bar_length.
        x_grid : np.ndarray | None, optional
            Optional common grid to evaluate all landscapes on.
        signed: bool
            If set to False, Deletes simplices apearing in both directions, corresponds to generalized landscapes with unsigned simplices

        Returns
        -------
        GeneralizedLandscapeFamily
        """

        if signed:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain.without_double_edges(), point_cloud))
    
            

        from forest_landscapes import compute_generalized_landscape_family

        return compute_generalized_landscape_family(
            self,
            _cycle_value,
            max_k=max_k,
            num_grid_points=num_grid_points,
            mode=mode,
            label=label,
            min_bar_length=min_bar_length,
            x_grid=x_grid,
            cache = cache
        )

    def plot_landscape_family(self, label: str,*args, **kwargs):
        from forest_landscapes import plot_landscape_family
        return plot_landscape_family(self, label,*args,**kwargs)

    def plot_landscape_comparison_between_functionals(self, labels: list[str],*args, **kwargs):
        from forest_landscapes import plot_landscape_comparison_between_functionals
        return plot_landscape_comparison_between_functionals(self, labels=labels, *args, **kwargs)

# --------- Animate comparison ------------

