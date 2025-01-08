import meshio
import sys
import numpy as np


'''
This file is just for testing purposes. It extracts all the points from a mesh file and saves them to a file.
 >>For the real application run the dealii program, which will save the right points of the selected mesh<<
'''
def extract_all_points(msh_file):
    # Read the .msh file
    mesh = meshio.read(msh_file)

    # Initialize a set to store unique points
    boundary_points = set()
    types = set()
    # Loop through cells (elements)
    for cell in mesh.cells:
        # Check for surface elements (triangles or quads)
        types.add(cell.type) 
        if cell.type in ["line", "triangle", "tetra"]:
            # Get the node indices for this element
            for element in cell.data:
                # Add each node index to the set
                boundary_points.add(tuple(mesh.points[element[0]])) 
                # for node_idx in element:
                #     print(mesh.points[node_idx])
                #     #print(mesh.points[node_idx]) 
                #     # ADD ALL POINTS IN THE NODE
                #     boundary_points.add(tuple(mesh.points[node_idx]))

    print(types)
    if "tetra" in types:
        dim = 3
    elif "triangle" in types:
        dim = 2
    else:
        dim = 1

    # Convert to numpy array
    boundary_points = np.array(list(boundary_points))
    boundary_points = boundary_points[:, :dim]

    return boundary_points

def save_points(points, output_file):
    """
    Save the boundary points to a file.

    Parameters:
        points (np.ndarray): Array of boundary points.
        output_file (str): Path to save the points.
    """
    np.savetxt(output_file, points, comments='', fmt="%.6f")

# Example usage
if __name__ == "__main__":
    # Path to the .msh file
    msh_file = "mesh-cube-40.msh"
    msh_file = sys.argv[1]
    #msh_file = "mesh-square-40.msh"

    # Extract boundary points
    all_points = extract_all_points(msh_file)
    print(f"Extracted {len(all_points)} unique mesh points")

    # Save the points to a file
    save_points(all_points, "mesh_points.txt")
    print("Points saved to 'mesh_points.txt'.")
