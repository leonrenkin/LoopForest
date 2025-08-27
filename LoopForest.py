from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Iterable, Set
import itertools
import numpy as np
import gudhi as gd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

# ------- helper functions --------------

def loop_in_filtration_check(loop, simplex_tree, filt_value):

    for i in range(len(loop)):
        birth = edge_birth_time(loop[i-1],loop[i],simplex_tree=simplex_tree)

        if math.isinf(birth) and birth > 0:
            print(f"Edge {[loop[i-1],loop[i]]} never appears in the filtration (birth = +∞).")
            return False
        
        elif birth > filt_value:
            print(f"Edge {[loop[i-1],loop[i]]} is in simplicial complex but appears later in the filtration")
            return False

    return True

def triangle_loop_counterclockwise(simplex, point_cloud):
    """
    Given `simplex`: a length-3 iterable of point indices into point_cloud,
    return those same indices reordered so that:
      1. The first point has the smallest (x, then y) coordinates.
      2. The remaining two follow in counter-clockwise order around that first point.
    """
    if len(simplex) != 3:
        raise ValueError("Error: given simplex is not a triangle.")

    # 1) Grab the actual coordinates as an (3,2) array
    simplex_arr     = np.array(simplex, dtype=int)
    triangle_pts    = np.asarray(point_cloud)[simplex_arr]   # shape (3,2)

    # 2) Find the sort order by x, then y
    #    lexsort takes keys in the order (secondary, primary), so:
    order = np.lexsort((triangle_pts[:,1], triangle_pts[:,0]))
    simplex_sorted = simplex_arr[order]
    pts_sorted     = triangle_pts[order]

    # 3) Compute the two edge-vectors out of the “lowest” point
    v1 = pts_sorted[1] - pts_sorted[0]
    v2 = pts_sorted[2] - pts_sorted[0]

    # 4) Use the sign of the 2D cross-product to check orientation:
    #    cross > 0 means v1→v2 is CCW; if it’s < 0, swap pts 1 and 2.
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0:
        simplex_sorted[1], simplex_sorted[2] = simplex_sorted[2], simplex_sorted[1]

    return simplex_sorted.tolist()

def merge_index_list_loops(loop1: List[int],loop2: List[int], edge:List[int]):
    """ 
    Merges two loops, given as list of indices, which both contain edge in oppossing orientation.
    Edge is pair of indices.
    Returns loop as list of indices. 
    """

    #find position of edge in first loop
    for i in range(len(loop1)):
        if loop1[i] == edge[0]:
            if not (loop1[(i-1) % len(loop1)]==edge[1] or loop1[(i+1) % len(loop1)]==edge[1]):  #deal with the case that one vertex in the edge appears an additional time
                #print("Concat loop edge case triggered")
                continue
            idx0= i
            break
    
    #check which edge appears first
    if loop1[idx0-1]== edge[1]: # type: ignore
        j = 0
        p = edge[1]
        idx0 = (idx0 - 1) % len(loop1) # type: ignore
    else:
        j = 1
        p = edge[0]
    
    for i in range(len(loop2)):
        if loop2[i]==p:
            if not (loop2[(i-1) % len(loop2)]==edge[j] or loop2[(i+1) % len(loop2)]==edge[j]):  #deal with the case that one vertex in the edge appears an additional time
                #print("Concat loop edge case triggered")
                continue
            idx1=i
            break

    #loop = loop1[:idx0] + loop2[idx1:] + loop2[:idx1-1] + loop1[ idx0  :]
    
    if idx0 == len(loop1)-1 and idx1==0: # type: ignore
        loop = loop1[1:]+ loop2[idx1+1:] + loop2[:idx1] # type: ignore
    elif idx0 == len(loop1)-1 and idx1!=0: # type: ignore #handle edge case that we have the last index for idx0
        loop = loop1+ loop2[idx1+1:] + loop2[:idx1-1]  # type: ignore
    else: #common case
        loop = loop1[:idx0] + loop2[idx1:] + loop2[:idx1] + loop1[ idx0+2  :] # type: ignore

    #print(F'resulting loop {loop}')
    if len(loop) != len(loop1)+len(loop2)-2:
        raise ValueError("Error, concatitnation has wrong length")

    return loop

def edge_birth_time(a, b, simplex_tree):
    # sort the endpoints so you get the same key for [a,b] or [b,a]
    edge = tuple(sorted((a, b)))
    return simplex_tree.filtration(edge)  # returns +inf if not present

def contains_pair_in_order(lst, a, b):
    """
    Returns True iff the list `lst` contains the adjacent subsequence [a, b].
    """

    for i in range(-1,len(lst)):
        if lst[i] == a and lst[i+1 ] == b:
            return True
    return False

def split_vertex_loop_with_double_edge(edge, loop):
    """splits a loop containg edge twice into two disjoint loops.
    return two loops, one of which might be empty"""
    if not (contains_pair_in_order(lst=loop, a=edge[0], b=edge[1]) and contains_pair_in_order(lst=loop, a=edge[1], b=edge[0])):
        raise ValueError("Input loop does not contain edge in both directions")
    for idx1 in range(len(loop)):
        if loop[idx1] in edge and (loop[idx1+1] in edge or loop[idx1-1] in edge):
            break
    if loop[idx1+1] in edge: # type: ignore #this is only to deal with the case idx = 0
        idx1 +=1 # type: ignore
    
    for idx2 in range(idx1+1, len(loop)): # type: ignore
        if loop[idx2] in edge and (loop[(idx2+1) % (len(loop)-1)] in edge or loop[idx2-1] in edge):
            break
    if idx2 !=len(loop)-1: # type: ignore
        idx2 +=1 # type: ignore

    #deal with edge case where we the edge [a,b] appears as [a,b,a]
    #in this case idx1 is at b and idx2 is the one after the second a 
    if idx2-idx1 == 2: # type: ignore
        print()
        loop1 = loop[:idx1] + loop[idx2:] # type: ignore
        loop2 = []
        #print(f"non-trivial loop returned: {loop1}")
        return loop1, loop2
    #same edge case but this time it is [a,....,a,b] or [b,a,....,a]
    if idx2 - idx1 == len(loop)-2: # type: ignore
        loop1 = loop[idx1:idx2] # type: ignore
        loop2 = []
        #print(f"non-trivial loop returned: {loop1}")
        return loop1, loop2


    loop1 = loop[idx1: idx2-1] # type: ignore
    
    if idx1 == 0: # type: ignore
        loop2 = loop[idx2:len(loop)-1]  # type: ignore
    
    else:
        loop2 = loop[idx2:] + loop[:idx1-1] # type: ignore

    #Loop is only a single vertex does not count as loop an will be treated as empty
    if len(loop1)==1:
        loop1=[] 
    if len(loop2)==1:
        loop2=[]

    #print(f"edge {edge}, loop {loop}")
    #print(f"idx1 {idx1}, idx2 {idx2}") # type: ignore
    #print(f"loop1 {loop1}")
    #print(f"loop2 {loop2}")


    return loop1, loop2

def loop_xmax(loop, point_cloud):
    return max(point_cloud[idx][0] for idx in loop)

def loop_xmin(loop, point_cloud):
    return min(point_cloud[idx][0] for idx in loop)

def loop_ymax(loop, point_cloud):
    return max(point_cloud[idx][1] for idx in loop)

def loop_ymin(loop, point_cloud):
    return max(point_cloud[idx][1] for idx in loop)

def find_outer_loop(vertex_loop:List[int] ,edge:List[int] ,point_cloud: List[List[float]]):
    """ Computes surviving loop in case of double edge in a single loop"""
    #print("tiebreak activated")
    loop1, loop2 = split_vertex_loop_with_double_edge(edge=edge, loop=vertex_loop)


    if len(loop1)<=2 and len(loop2)<=2:
        raise ValueError("Both split loops are trivial")
    elif len(loop1) <=2:
        return loop2

    elif len(loop2) <=2:
        return loop1
    
    else:
        #check which loop is contained by the other one
        xmax1=loop_xmax(loop=loop1, point_cloud=point_cloud)
        xmax2=loop_xmax(loop=loop2, point_cloud=point_cloud)

        #could use ymax,  ymin, xmin to check if htey also satisfy the inequalities
        if xmax1>xmax2:
            return loop1

        if xmax1<xmax2:
            return loop2

        else:
            raise ValueError("Maximal x value of both loops is equal which should not happen")

    
# ----------------- class defintions ----------------------
@dataclass
class Loop:
    """
    Loop saved as a list of vertex indices. 
    Point coordinates can be accessed via point_cloud[vertex] where vertex is in vertex_list and point_cloud is the list of points saved in the loop forest
    """
    vertex_list: List[int]
    id: int
    active_start: float = -1           #[active_start, active_end) is interval in which the loop is active as an optimal cycle rep
    active_end: float = -1              #-1 as default since this should not appear in computations


@dataclass
class Node:
    """ Objects which are the nodes in the LoopForest graph. 
    Each node has a loop representative."""
    id: int
    filt_val: float
    type: Literal["leaf", "root", "merge", "update"]            #Special case if points are not in general position: a node can be type "merge" and also be a root, it then still appears in the root list
    loop: Loop                                                  #Loops are saved as list of indices of simplex
    children: Tuple[int,...]                                    #ids of children
    parent: Optional[int] = None
    #is_root: bool = True  #True if it is the root of a tree, also used for bookkeeping of active loops

    def __repr__(self) -> str:
        return f"Node(id={self.id}, f={self.filt_val}, type={self.type})"
    
@dataclass
class Bar:
    """ 
    Object which stores a bar in H1 persistence together with a progression of cycle reps.
    Each cycle rep has a an active_start and active_end attribute which is the interval in which this representative is optimal.
    The cycle reps are a strictly decreasing chain w.r.t. inclusion.
    """
    birth: float
    death: float
    _node_progression: tuple[int,...] #nodes saved as node_ids
    cycle_reps: list[Loop]


class LoopForest:
    """Object that stores progression of optimal loops for alpha complex of a point cloud in a forest format"""

    def __init__(self, 
                 point_cloud: List[List[float]]) -> None:
        self.point_cloud = point_cloud #point cloud is list of 2-dim arrays

        self._node_id = itertools.count(1)         #used to assign unique id to each node in the forest
        self._loop_id = itertools.count(1)
        self.nodes: Dict[int, Node] = {}
        self.loops: Dict[int, Loop] = {}

        self._active_node_ids: Set[int] = set()               #used for bookkeeping of active nodes in forest computation algorithm
        self.roots: List[Node]  = []                    #List of roots of trees in the forest

        self.levels: list[float] = []               #Critical filtration values, might give duplicates in current implementation

        self.alpha_complex = gd.AlphaComplex(points=point_cloud) # pyright: ignore[reportAttributeAccessIssue]
        self.simplex_tree = self.alpha_complex.create_simplex_tree()

        # Extract s filtration up to order 2
        self.filtration = [f for f in self.simplex_tree.get_filtration() if len(f[0]) <= 3]  # Keep simplices up to 2D

        self.barcode: List[Bar] = []



    # ---------- builders ---------

    def generate_loop(self,vertex_list): #necessary to generate id for the loop
        lid= next(self._loop_id)
        return Loop(vertex_list=vertex_list, id=lid)

    def add_leaf(self, triangle: List[int], filt_val: float):
        """
        Create a new leaf in tree. 
        Corresponds to death of a loop in homology and adding a loop in algorithm for computing the LoopForest.
        """

        nid = next(self._node_id)
        lid = next(self._loop_id)

        triangle_counterclockwise = triangle_loop_counterclockwise(simplex=triangle, point_cloud=self.point_cloud) #orient vertices in triangle counterclockwise
        new_loop = Loop(vertex_list=triangle_counterclockwise, id=lid)

        new_node = Node(id=nid,filt_val=filt_val, type='leaf', children=tuple(), loop=new_loop)

        self.nodes[nid]=new_node
        self.loops[lid]= new_loop
        self._active_node_ids.add(nid)  #roots are active nodes in algorithm, at termination the active nodes are precicely the roots of the forest 

        self.levels.append(filt_val)

        return
    
    def make_root(self, node: Node, filt_val: float):
        """
        Ends this tree in the forest by creating the root as a top node.
        Corresponds to birth in homology and removing an edge of a loop without merging it with another loop in LoopForest computation algorithm.
        """
        nid = next(self._node_id)

        self._active_node_ids.remove(node.id)
        node.parent = nid #new root node is parent of input node

        root_node = Node(id=nid, filt_val=filt_val,type='root', loop=node.loop, children=(node.id,))


        self.nodes[nid]=root_node
        self.roots.append(root_node)

        self.levels.append(filt_val)

        return

    def merge_nodes(self, node1: Node, node2: Node, parent_loop: Loop, filt_val: float):
        """ 
        Creates parent node of node1 and node2 with loop representative parent loop
        Corresponds to a loop being split into two loops in homology and a new bar appearing in barcode
        """

        nid = next(self._node_id)
        node1.parent = nid
        node2.parent = nid

        parent_node = Node(id=nid, filt_val=filt_val, type="merge", children=(node1.id,node2.id), loop = parent_loop)
        self.nodes[nid]=parent_node
        
        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node1.id)
        self._active_node_ids.remove(node2.id)

        self.levels.append(filt_val)

        return
    
    def update_node(self, node: Node, updated_loop: Loop, filt_val:float):
        """ Updates loop representative, corresponds to node with one parent and one child in tree """

        nid = next(self._node_id)
        node.parent=nid

        update_node = Node(id=nid, filt_val=filt_val, type="update", children=(node.id,),loop=updated_loop)
        self.nodes[nid]=update_node

        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node.id)

        self.levels.append(filt_val)

        return

    def merge_loops(self, edge: List[int], loop1: Loop, loop2: Loop) -> Loop:
        """ Merges two loops which both contain edge in opposing orientations"""

        nid = next(self._loop_id)

        loop_vertex_list = merge_index_list_loops(loop1 = loop1.vertex_list,loop2 = loop2.vertex_list, edge=edge)

        merged_loop = Loop(id=nid, vertex_list=loop_vertex_list)
        self.loops[nid] = merged_loop

        return merged_loop
    

    # ---- helper functions ------
    def nodes_with_loop_containing_edge(self ,edge: List[int], node_ids: Set[int]) -> List[Node]:
        """
        Input: List of node ids and edge as pair of indexes

        Finds nodes where the associated loop contains the edge. Duplicates are possible if the edge appears in a loop multiple time, i.e., the node will be listed twice.

        returns list of nodes in one of the following forms: [],[l1],[l1,l2],[l1,l1]
        """

        nodes = [self.nodes[id] for id in node_ids]

        #list of nodes containing the edge "simplex"
        node_list=[]

        if len(edge)!=2:
            raise ValueError("Error, simplex is not an edge")
            return

        #check which loops contain simplex
        for node in nodes:
            vertex_loop = node.loop.vertex_list
            for i in range(len(vertex_loop)):
                if {vertex_loop[i], vertex_loop[(i+1) %  len(vertex_loop)]} == {edge[0], edge[1]}:
                    node_list.append(node)

        if len(node_list)>2:
            raise ValueError(f"Error, too many loops contain edge {edge}")

        return node_list
    

    # ----- methods to work with the forest

    def active_nodes_at(self, filt_val: float) -> List[Node]:
        """
        Return list of active nodes at a given filtration value.

        A node is active at r if:
        - node.filt_val >= r
        - has a parent and parent has filt val < r
        """
        nodes = self.nodes

        active: List[Node] = []
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

    def leaves_below_node(self, node: Node) -> set[int]:
        """ Returns set of all leaves below a given node (below in tree means higher filtration value) """
        leaf_ids: set[int] = set()

        if node.type == "leaf":
            leaf_ids.add(node.id)
            return leaf_ids
        

        if len(node.children) == 0:
            raise ValueError(f"Node {node} is not a leaf but has no children, this should not happen")
        
        for cid in node.children:
            child = self.nodes[cid]
            leaf_ids.update(self.leaves_below_node(child))


        return leaf_ids

    def leaf_to_node_path(self, leaf: Node, node: Node) -> List[int]:
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

    def node_to_leaf_path(self, leaf: Node, node: Node) -> List[int]:
        """ 
        Returns direct path from a node to a leaf.
        Path is returned as list of node ids

        If there does not exist a path, an error is raised.
        """
        return list(reversed(self.leaf_to_node_path(leaf=leaf, node=node)))

    # ----- reduce forest (collapses trivial edges which happen at the same filtration value) -------------

    def _collapse_parent_child(self, parent: Node, child: Node):
        """
        Collapses a parent - child pair into the parent node.
        Intended for parent - child pair with same filtration value 
        """
        #print("collapsing parent and child")

        #collect new children of parents
        new_children = [cid for cid in parent.children if cid != child.id] #add children of parent apart from the child we collapse
        for gcid in child.children:
            new_children.append(gcid)

        parent.children = tuple(new_children)

        #re-parent grandchildren
        for gcid in child.children:
            grandchild = self.nodes[gcid]
            grandchild.parent = parent.id

        #remove child node from forest
        del self.nodes[child.id]


        #adapt type of parent node
        if len(parent.children)==0:
            parent.type="leaf"

            #if parent is now isolated point in forest, delete it from forest completely
            if parent.parent == None:
                del self.nodes[parent.id]
                self.roots.remove(parent)

        elif len(parent.children)==1:
            parent.type="update"
        else:
            parent.type="merge"


        return

    def _reduce_forest(self):
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
        print("Reducing the forest")

        changed = True 
        while changed: #in practice, this loop runs only once since we iterate over children while adding new children
            changed = False
            #print("reducing loop started")

             #iterate over snapshot of the nodes in the tree, the nodes dict might be changed each iteration
            node_list = list(self.nodes.values())
            for p in node_list:

                # p might have been deleted as a child in a previous iteration of this outer loop
                if p.id not in self.nodes.keys():
                    continue

                for cid in p.children:
                    child = self.nodes[cid]
                    if child.filt_val == p.filt_val:
                        self._collapse_parent_child(parent=p, child=child)
                        changed = True

        #sanity check to make sure parent pointers are correct, should no be necessary
        for n in self.nodes.values():
            for cid in n.children:
                ch = self.nodes.get(cid)
                if ch is None:
                    continue
                if ch.parent != n.id:
                    ch.parent = n.id
                    print(f"Parent pointer of node with id {cid} was retroactively fixed "
                        f"(was {ch.parent}, set to {n.id}).")

        print("Reduction complete")

        return

    # If we have multiple edges appearing at the same filtration value, we might get a root node which is also a merge in the reduction process
    # This will lead to a node which appears in the root list but has type merge
    # Not a mistake in the code, simply the way the edge case is currently handled
    # -> use node.parent == None to check if a node is a root

    # ------ Add active period of each loops ----------

    def _compute_loop_activity(self):
        """ Computes period in which each loop is an optimal cycle rep and adds it to the loop as attributes """

        for node in self.nodes.values():

            if node.parent == None:
                continue
            
            parent = self.nodes[node.parent]
            node.loop.active_end = node.filt_val
            node.loop.active_start = parent.filt_val


        return

    # ----- Compute barcode sequence ---------

    def _compute_tree_barcode(self, node: Node, child_id: int):
        """ Recursively computes barcode of sub-tree below the input node and the child with child_id"""

        #compute longest bar.
        choosen_child = self.nodes[child_id]
        leaf_ids = self.leaves_below_node(node=choosen_child)

        max_leaf_id = max( leaf_ids, key = lambda id: self.nodes[id].filt_val)
        max_leaf = self.nodes[max_leaf_id]

        path = self.node_to_leaf_path(node=node, leaf = max_leaf)
        if len(path)<=1:
            raise ValueError("path for barcode too short, something went wrong")

        cycle_reps = [self.nodes[id].loop for id in path[1:]]
        
        bar = Bar(birth=node.filt_val, death=max_leaf.filt_val, _node_progression=tuple(path), cycle_reps=cycle_reps )
        self.barcode.append(bar)

        #print(f"{bar} added")

        #at every merge node, compute barcode of subtree of merge nodes with the other children
        for id in self.leaf_to_node_path(node=node,leaf=max_leaf)[:-2]:   #we do not want to repeat the top node of the tree
            child_node = self.nodes[id]
            pid = child_node.parent
            if pid == None:
                raise ValueError(f"Parent node incounter, this should not happen. Node id {pid}")
            parent_node = self.nodes[pid]

            if parent_node.type == "merge":
                for cid in parent_node.children:
                    if cid == id:
                        continue 
                    else:
                        self._compute_tree_barcode(node=parent_node, child_id=cid)      

    def compute_barcode(self):
        """ Computes H1 barcode of forest and stores it in self.barcode """
        print("Computing Barcode")
        #compute barcode for each tree
        for root in self.roots:
            for child_id in root.children:
                self._compute_tree_barcode(node=root, child_id=child_id)
        print("Barcode computation completed")

        return

    
    # ----- plotting tools -------

    #ChatGPT plotting function
    def plot_at_filtration( 
        self,
        filt_val: float,
        ax=None,
        show: bool = True,
        fill_triangles: bool = True,
        loop_vertex_markers: bool = False,
    ):
        """
        Plot the 2-D point cloud, all edges/triangles with filtration <= filt_val,
        and overlay the loops of the nodes active at filt_val.

        Notes
        -----
        - GUDHI's AlphaComplex / SimplexTree work in α² units (squared radius).
        Pass the same units here.
        - Uses SimplexTree.get_filtration(), which is sorted by increasing filtration.

        Parameters
        ----------
        filt_val : float
            Filtration threshold (α² units).
        ax : matplotlib.axes.Axes or None
            Axes to draw on; if None, a new figure+axes are created.
        show : bool
            If True, calls plt.show() when done.
        fill_triangles : bool
            If True, lightly fill triangles present at this filtration.
        loop_vertex_markers : bool
            If True, mark the vertices used in each loop.

        Returns
        -------
        matplotlib.axes.Axes
        """
        

        # --- Prep
        pts = np.asarray(self.point_cloud, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("point_cloud must be an (n_points, 2) array-like.")

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))

        st = self.simplex_tree

        # --- Collect edges and triangles present at this filtration value
        edges_xy = []      # list of [[x1,y1],[x2,y2]]
        tris_xy = []       # list of [[x1,y1],[x2,y2],[x3,y3]]
        for simplex, f in st.get_filtration():
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
        ax.scatter(pts[:, 0], pts[:, 1], s=18, color="k", zorder=3, label="points")

        # --- Draw triangles first (under edges)
        if fill_triangles and tris_xy:
            tri_coll = PolyCollection(
                tris_xy, closed=True, edgecolors="none", facecolors="C0", alpha=0.15, zorder=1
            )
            ax.add_collection(tri_coll)

        # --- Draw edges
        if edges_xy:
            edge_coll = LineCollection(edges_xy, linewidths=1.0, colors="0.65", zorder=2, label="edges")
            ax.add_collection(edge_coll)

        # --- Overlay loops from active nodes at filt_val
        active = self.active_nodes_at(filt_val)
        cmap = plt.get_cmap("tab10")
        for idx, node in enumerate(active):
            if node.loop is None or not getattr(node.loop, "vertex_list", None):
                continue
            vs = list(node.loop.vertex_list)
            if len(vs) < 2:
                continue

            # Build closed polyline for the loop
            closed_vs = vs + [vs[0]]
            loop_xy = pts[closed_vs]  # shape: (m, 2)

            # >>> make a Sequence[ArrayLike] (list of 2x2 arrays) for Pylance
            segments = [np.array([loop_xy[i], loop_xy[i + 1]]) for i in range(len(loop_xy) - 1)]

            # Thicker colored edges along the loop
            loop_coll = LineCollection(segments, linewidths=2.5, colors=["orange"], zorder=5)
            ax.add_collection(loop_coll)

            # Optional vertex markers for the loop
            if loop_vertex_markers:
                ax.scatter(
                    pts[vs, 0], pts[vs, 1],
                    s=36, color="orange", edgecolors="white", linewidths=0.8, zorder=6
                )

        # --- Aesthetics
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"α² ≤ {filt_val:.4g}  •  edges/triangles in filtration + active loops")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # A simple legend (points + edges); loop colors are self-explanatory on top
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower right", frameon=True)

        ax.autoscale()  # fit collections
        if show:
            plt.show()
        return ax

    #ChatGPT plotting function
    def plot_dendrogram(
        self,
        ax=None,
        show: bool = True,
        annotate_ids: bool = False,
        leaf_spacing: float = 1.0,
        tree_gap_leaves: int = 1,
        check_reduced: bool = True,
        small_on_top: bool = False,
    ):
        import warnings

        if not self.nodes:
            raise ValueError("LoopForest has no nodes to plot.")

        nodes = self.nodes
        node_ids = set(nodes.keys())

        # Recompute roots robustly from the current structure (these are Node objects)
        roots = [n for n in nodes.values() if n.parent is None or n.parent not in node_ids]
        if not roots and getattr(self, "roots", None):
            roots = list(self.roots)

        if check_reduced:
            equal_pairs = [
                (p.id, c)
                for p in nodes.values()
                for c in p.children
                if c in nodes and nodes[c].filt_val == p.filt_val
            ]
            if equal_pairs:
                warnings.warn(
                    f"Forest does not appear reduced: found {len(equal_pairs)} parent–child pairs "
                    f"with equal filt_val. Plotting anyway."
                )

        bad_direction = [
            (p.id, c)
            for p in nodes.values()
            for c in p.children
            if c in nodes and nodes[c].filt_val < p.filt_val
        ]
        if bad_direction:
            warnings.warn(
                f"{len(bad_direction)} edges have child.filt_val > parent.filt_val."
            )

        roots.sort(key=lambda n: (n.filt_val, n.id))

        # --- positions
        x: dict[int, float] = {}
        y: dict[int, float] = {n.id: float(n.filt_val) for n in nodes.values()}
        visited: set[int] = set()
        leaf_counter = 0

        def _assign_x(nid: int):
            nonlocal leaf_counter
            if nid in visited:
                return
            visited.add(nid)

            child_ids = [cid for cid in nodes[nid].children if cid in nodes]
            child_ids.sort(key=lambda cid: (nodes[cid].filt_val, nodes[cid].id))

            if len(child_ids) == 0:
                x[nid] = leaf_counter * leaf_spacing
                leaf_counter += 1
            else:
                for cid in child_ids:
                    _assign_x(cid)
                xs = [x[cid] for cid in child_ids]
                x[nid] = sum(xs) / len(xs)

        # Lay out each tree; insert spacing between trees
        for i, r in enumerate(roots):
            start_before = leaf_counter
            _assign_x(r.id)  # <-- pass the integer id
            if i != len(roots) - 1 and leaf_counter > start_before:
                leaf_counter += tree_gap_leaves

        # Place any stray components (shouldn't happen, but be safe)
        for nid in list(node_ids):
            if nid not in x:
                _assign_x(nid)
                leaf_counter += tree_gap_leaves

        # --- draw
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        line_color = "0.25"
        merge_color = "0.1"
        node_edge = "white"

        for p in nodes.values():
            pid = p.id
            px, py = x[pid], y[pid]
            child_ids = [cid for cid in p.children if cid in nodes]
            if not child_ids:
                continue

            child_ids.sort(key=lambda cid: x[cid])
            child_xs = [x[cid] for cid in child_ids]
            child_ys = [y[cid] for cid in child_ids]

            for cx, cy in zip(child_xs, child_ys):
                ax.plot([cx, cx], [cy, py], linewidth=1.2, color=line_color, zorder=1)

            if len(child_ids) >= 2:
                ax.plot([min(child_xs), max(child_xs)], [py, py], linewidth=1.5, color=merge_color, zorder=2)

        for n in nodes.values():
            ax.scatter(x[n.id], y[n.id], s=24, zorder=3, color="C0", edgecolors=node_edge, linewidths=0.6)
            if annotate_ids:
                ax.annotate(
                    str(n.id),
                    (x[n.id], y[n.id]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    color="0.2",
                )

        ax.set_xlabel("leaf order")
        ax.set_ylabel("filt_val (y)")
        ax.set_title("LoopForest dendrogram (y = filt_val)")
        ax.margins(x=0.05, y=0.05)
        ax.grid(False)

        if x:
            xs = list(x.values())
            ax.set_xlim(min(xs) - leaf_spacing, max(xs) + leaf_spacing)

        if small_on_top:
            ax.invert_yaxis()

        if show:
            plt.show()
        return ax
    
    #ChatGPT plotting function
    def plot_barcode(
        self,
        *,
        ax=None,
        sort: str | None = "length",   # "length" | "birth" | "death" | None
        title: str = "Barcode",
        xlabel: str = "filtration value",
    ):
        """
        Plot a 1D barcode from self.barcode (a list[Bar]).
        Each Bar contributes a horizontal segment from birth to death.
        If death is +inf, an arrow is drawn to the right.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            If given, draw on this axes. Otherwise a new figure/axes is created.
        sort : {"length","birth","death",None}
            Sort bars before plotting (None preserves current order).
        title : str
            Plot title.
        xlabel : str
            Label for the x-axis.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the barcode was drawn on.
        """
        import math
        import numpy as np
        import matplotlib.pyplot as plt

        if not getattr(self, "barcode", None):
            raise ValueError("No bars to plot: `self.barcode` is empty.")

        # Work on a copy so we don't mutate user order unless requested.
        bars = list(self.barcode)

        # Optional sorting
        if sort == "birth":
            bars.sort(key=lambda b: (b.birth, b.death))
        elif sort == "death":
            def dkey(b):
                return (math.inf if not math.isfinite(b.death) else b.death, b.birth)
            bars.sort(key=dkey)
        elif sort == "length":
            def length(b):
                return (math.inf if not math.isfinite(b.death) else b.death) - b.birth
            bars.sort(key=length, reverse=True)

        # Create axes if needed
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, max(2.0, 0.35 * len(bars))))
            created_ax = True
        else:
            fig = ax.figure

        # Determine x-limits with a bit of padding
        births = np.array([b.birth for b in bars], dtype=float)
        deaths = np.array([b.death for b in bars], dtype=float)
        finite_deaths = deaths[np.isfinite(deaths)]

        xmin = float(np.nanmin(births))
        xmax = float(np.nanmax(finite_deaths)) if finite_deaths.size else float(np.nanmax(births))
        if not np.isfinite(xmax):  # extreme corner case
            xmax = xmin

        pad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
        ax.set_xlim(xmin - pad, xmax + pad)

        # Draw segments
        for i, b in enumerate(bars):
            x0, x1 = float(b.birth), float(b.death)

            # Guard against inverted bars due to numerical issues
            if math.isfinite(x1) and x1 < x0:
                x0, x1 = x1, x0

            if math.isfinite(x1):
                ax.hlines(y=i, xmin=x0, xmax=x1, linewidth=2)
                # Optional end ticks (comment out if you don't want them)
                ax.plot([x0, x1], [i, i], linestyle="None", marker="|", markersize=8)
            else:
                # Draw to the right with an arrow for infinity
                right = ax.get_xlim()[1]
                ax.hlines(y=i, xmin=x0, xmax=right - 0.25 * pad, linewidth=2)
                ax.annotate(
                    "",
                    xy=(right - 0.15 * pad, i),
                    xytext=(x0, i),
                    arrowprops=dict(arrowstyle="->", lw=2),
                    va="center"
                )

        # Cosmetics
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        fig.tight_layout()

        # If we created the axes, show it immediately (so this works in scripts)
        if created_ax:
            import matplotlib.pyplot as plt
            plt.show()

        return ax


#------------ helper functions which use classes ---------------



# ---------- Compute Loop Forest -----------------
def compute_loop_forest(point_cloud, reduce: bool = True, compute_barcode= True):
    """ 
    Computes LoopForest object for a point cloud.
    reduce = True means that multiple changes at the same filtration value is collapsed to a single node.
    compute_barcode = True computes barcodes and stores it in self.barcode as list of bar objects
    """
    loop_forest = LoopForest(point_cloud=point_cloud)
    
    #simplices is already ordered in ascending order by number of simplices 
    #print(loop_forest.filtration)

    for simplex, filt_val in reversed(loop_forest.filtration):

        if len(simplex)<=1:
            continue
        
        #print("\nstarting with" ,simplex, filt_val, ", r=", filt_val)
        #
        # print(f"active node ids = {loop_forest._active_node_ids}")
        
        if len(simplex) == 3:
            loop_forest.add_leaf( triangle=simplex, filt_val=filt_val)

        if len(simplex) == 2:
            #L is loops containing nodes, can be of the form [],[l1], [l1,l2], [l1,l1]
            L = loop_forest.nodes_with_loop_containing_edge(edge = simplex, node_ids=loop_forest._active_node_ids)

            #if no loop contains edge, nothing happens
            if len(L) == 0:
                continue

            #if edge is only contained in a single loop and appears only once in that loop once, remove that loop from the active loops 
            elif len(L) == 1:
                loop_forest.make_root(node=L[0],filt_val=filt_val)
                continue

            elif len(L) == 2 and L[0]!=L[1]:
                parent_loop = loop_forest.merge_loops(loop1=L[0].loop ,loop2= L[1].loop, edge=simplex) 
                loop_forest.merge_nodes( node1=L[0], node2=L[1], parent_loop=parent_loop, filt_val=filt_val)
                if not loop_in_filtration_check(parent_loop.vertex_list, simplex_tree=loop_forest.simplex_tree, filt_value=filt_val):
                        raise ValueError("Loop not in simplex, Loop concat Case")


            elif len(L) == 2 and L[0]==L[1]:
                #Same edge is contained in a loop in both directions, we update the loop
                vertex_loop = find_outer_loop(edge=simplex,
                                              vertex_loop=L[0].loop.vertex_list, 
                                              point_cloud=loop_forest.point_cloud)
                if not loop_in_filtration_check(vertex_loop, simplex_tree=loop_forest.simplex_tree, filt_value=filt_val):
                            print('edge', simplex)
                            print(f'starting loop', L[0].loop)
                            print(f'outer loop', vertex_loop)
                            raise ValueError("Loop not in simplex, Tiebreak Case")
                
                updated_loop = loop_forest.generate_loop(vertex_loop)

                loop_forest.update_node(node=L[0], updated_loop=updated_loop, filt_val=filt_val)

            else:
                raise ValueError("Error, L is of the wrong form")

    print("Forest succesfully computed")

    loop_forest._compute_loop_activity()

    if reduce:
        loop_forest._reduce_forest()

    if compute_barcode:
        loop_forest.compute_barcode()
        

    return loop_forest



