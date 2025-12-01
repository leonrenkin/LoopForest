import numpy as np
import matplotlib.pyplot as plt


def generate_point_cloud_2d(min=0,max=1,save_to_csv=False, filename="point_cloud.csv",timeout = 60):
    """
    interactive way to generate 2d pointcloud in [-1,1] x [-1,1] by simply clicking to add points
    returns array of 2dim tuples
    min and max specify interval boundaries
    """

    # Initialize an empty list to store points
    point_cloud=[]
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_title("Click to add points. Close the window when done.")
    ax.set_xlim(min, max)  # Set X-axis limits
    ax.set_ylim(min, max)  # Set Y-axis limits
    ax.set_aspect("equal")

    # Use ginput to collect points
    print("Click on the plot to add points. Close the plot window when done.")
    point_cloud = plt.ginput(n=-1, timeout=timeout)  # n=-1 allows unlimited clicks; timeout=0 waits indefinitely
    plt.show()
    plt.close()
    # Save to CSV if the option is enabled
    if save_to_csv:
        # Use NumPy for efficient saving
        np.savetxt(filename, point_cloud, delimiter=",", header="x,y", comments="")
        print("Point cloud saved to", filename)

    # Convert point_cloud from a list of arrays to a list of tuples (tuples are more efficient than arrays)
    point_cloud = [(p[0], p[1]) for p in point_cloud]

    return point_cloud


filename="tmp.csv"

#manually generate point cloud
axis_min=0
axis_max=1
point_cloud_1=generate_point_cloud_2d(axis_min,axis_max,save_to_csv=True, filename=filename,timeout=90)
