# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from PersistenceForest import PersistenceForest


# %%
rng = np.random.default_rng(35)
num_points=300
points = rng.uniform(low=0.0, high=2*np.pi, size=num_points)
points = np.sqrt(np.abs(np.cos(1.5*points))+.1)[:,None] * np.column_stack((np.cos(points), np.sin(points))) + rng.normal(scale=0.05, size=(num_points,2))

plt.figure(figsize=(6,6))
plt.scatter( points[:,0], points[:,1], s = 3)
plt.axis('equal')

pers_forest = PersistenceForest(points, print_info=True)

# %%
pers_forest.plot_barcode(min_bar_length=0.01, coloring = "forest")

# %%
pers_forest.plot_at_filtration(0.2)
pers_forest.plot_at_filtration(0.6)
pers_forest.plot_at_filtration(0.7)
# %%
from cycle_rep_vectorisations import signed_chain_edge_length, constant_one_functional

pers_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_edge_length,
    max_k=6,
    num_grid_points=1000,
    label="length",
)

pers_forest.compute_generalized_landscape_family(
    cycle_func=constant_one_functional,
    max_k=6,
    num_grid_points=1000,
    label="1",
)

pers_forest.plot_landscape_family(label='length', title = "Length Persistence Landscapes")
pers_forest.plot_landscape_family(label="1", title = "Regular Persistence Landscapes")


# %%
from cycle_rep_vectorisations import signed_chain_excess_connected_components, signed_chain_area, signed_chain_connected_components, signed_chain_connected_components, signed_chain_excess_connected_components

double_edge_cloud = np.loadtxt("point_cloud_csvs/signed_chain_example.csv",  delimiter=",", skiprows=1) * 100
double_edge_forest = PersistenceForest( point_cloud=double_edge_cloud )

double_edge_forest.plot_at_filtration(15,loop_edge_arrows=True)
signed_cycle = double_edge_forest.max_bar().cycle_at_filtration_value(15)


# %%

double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="signed components",
)

double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_excess_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="signed excess components",)

double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="unsigned components",
    signed=False)    #signed is True by default, set signed=False for unsigned versions

double_edge_forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_excess_connected_components,
    max_k=6,
    num_grid_points=1000,
    label="unsigned excess components",
    signed=False)   

double_edge_forest.plot_landscape_comparison_between_functionals(labels=["signed components", "unsigned components"] )
double_edge_forest.plot_landscape_comparison_between_functionals(labels=["signed excess components", "unsigned excess components"] )


# %%
from forest_landscapes import MultiLandscapeVectorizer
# 1. define functions
cycle_funcs = [signed_chain_edge_length]

# 2. Collect LoopForest objects
forests_train = [pers_forest]  # list of LoopForest, these need to be computed depending on the given task you are working in
forests_test  = [pers_forest]

# 3. Create the vectoriser
vec = MultiLandscapeVectorizer(
    cycle_funcs=cycle_funcs,
    max_k=3,
    num_grid_points=64,
    min_bar_length=0.01,
    include_stats=True,  # adds L1/L2 norms per level
)

# 4. Fit on training forests (learns common grid)
vec.fit(forests_train)

# 5. Vectorise datasets
X_train = vec.transform(forests_train)
X_test  = vec.transform(forests_test)

# 6. Use X_train / X_test in any ML model
from sklearn.ensemble import RandomForestClassifier
y_train = [0] #set categories for forest_train/X_train (rightn now this is nonsense)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# %%
