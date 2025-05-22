import subprocess
import os

# --- Configuration ---
GMSH_EXE_PATH = "gmsh"  # Or provide the full path if not in PATH, e.g., "/usr/local/bin/gmsh"
DEFAULT_LC_GLOBAL = 1.0  # Default characteristic length for Gmsh if not calculable
LC_RATIO_BOUNDARY = 0.03 # Target mesh size as a ratio of the characteristic length of the domain
LC_RATIO_HOLE = 0.015    # Finer mesh for holes

def parse_obj(obj_filepath):
    """Parses an OBJ file to extract 2D vertices and line segments."""
    vertices_3d = []  # Store original 3D vertices to maintain OBJ indexing
    vertices_2d = []  # Store (y, z) as (x_2d, y_2d)
    edges = []  # Store pairs of 1-based vertex indices

    print(f"Parsing OBJ file: {obj_filepath}")
    with open(obj_filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            
            cmd = parts[0]
            if cmd == 'v':
                try:
                    # x = float(parts[1]) # Ignored
                    y = float(parts[2])
                    z = float(parts[3])
                    vertices_3d.append((float(parts[1]), y, z)) # Keep original for reference
                    vertices_2d.append((y, z)) # Use (y,z) as our 2D (x,y)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed vertex line {line_num}: {line.strip()} - {e}")
            elif cmd == 'l':
                try:
                    # OBJ is 1-indexed
                    v1_idx = int(parts[1])
                    v2_idx = int(parts[2])
                    edges.append((v1_idx, v2_idx))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line segment line {line_num}: {line.strip()} - {e}")
    
    print(f"Found {len(vertices_2d)} vertices and {len(edges)} edges.")
    if not vertices_2d:
        raise ValueError("No vertices found in OBJ file.")
    # No check for edges being empty, as Gmsh can work with just points if needed for other things.
    # But for this task, edges are essential for line loops.
    if not edges:
        print("Warning: No line segments ('l' commands) found in OBJ file. Cannot form loops.")

    return vertices_2d, edges, vertices_3d

def calculate_polygon_area(polygon_vertices):
    """Calculates the signed area of a 2D polygon given its vertices in order.
       Positive if CCW, negative if CW.
    """
    area = 0.0
    n = len(polygon_vertices)
    if n < 3:
        return 0.0
    for i in range(n):
        x1, y1 = polygon_vertices[i]
        x2, y2 = polygon_vertices[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return area / 2.0

def trace_polygons(num_vertices, edges_1_based):
    """
    Traces closed polygons (loops) from a list of edges.
    Returns a list of polygons, where each polygon is a list of 0-based vertex indices.
    """
    print("Identifying contours (tracing polygons)...")
    adj = [[] for _ in range(num_vertices)]
    for v1_obj, v2_obj in edges_1_based:
        # Convert 1-based OBJ indices to 0-based Python indices
        u, v = v1_obj - 1, v2_obj - 1
        if 0 <= u < num_vertices and 0 <= v < num_vertices:
            adj[u].append(v)
            adj[v].append(u)
        else:
            print(f"Warning: Edge ({v1_obj}, {v2_obj}) references out-of-bounds vertex index. Max index: {num_vertices-1}. Skipping edge.")


    # Make sure adjacency lists are unique (though usually they are by construction if edges are unique)
    for i in range(num_vertices):
        adj[i] = list(set(adj[i]))

    polygons_0_based = []
    visited_edges = set() # To avoid re-tracing edges in complex scenarios (though simple loop tracing might not need it explicitly)
    visited_nodes_in_current_polygon_trace = [False] * num_vertices # For the current trace
    
    # More robust tracing: keep track of globally visited nodes to start new polygons
    globally_visited_nodes = [False] * num_vertices

    for start_node_idx in range(num_vertices):
        if not globally_visited_nodes[start_node_idx] and adj[start_node_idx]:
            # Try to trace a polygon starting from start_node_idx
            current_polygon = [start_node_idx]
            current_node = start_node_idx
            
            # Pick an initial direction
            if not adj[current_node]: continue # Isolated node
            
            # Try to pick a neighbor that hasn't been part of another polygon yet, if possible.
            # This is tricky without more complex graph analysis. For now, simple next.
            prev_node = -1 # No previous node initially for choosing first step

            # Find first step:
            # Try to pick a neighbor that isn't globally visited yet, if any.
            # Otherwise, just pick the first one.
            
            # This simple tracing might still have issues with complex junctions if a node
            # is part of multiple "conceptual" boundary lines that are not distinct loops.
            # The MATLAB version had a slightly more complex loop for tracing.

            # Simplified loop tracing: always try to move to an unvisited neighbor (in this path)
            # that is not the one we just came from.

            path_found = False
            
            # We need to be careful about picking the "next" node to continue the loop consistently.
            # A common approach is to sort neighbors by angle if you want canonical tracing,
            # but for just finding loops, picking any valid one works.

            q = [(start_node_idx, [start_node_idx])] # (current_node, path_so_far)
            
            # This BFS-like approach is for finding a single loop. We need something iterative.
            # Let's stick to a sequential path tracing.

            # Reset for each new potential polygon start
            current_path = [start_node_idx]
            cn = start_node_idx
            pn = -1 # Previous node in path

            while len(current_path) <= num_vertices : # Max path length safety
                
                possible_next_nodes = []
                for neighbor in adj[cn]:
                    if neighbor == pn: # Don't go back immediately
                        continue
                    possible_next_nodes.append(neighbor)
                
                if not possible_next_nodes: # Dead end
                    break 

                # If start_node_idx is a possibility, and path is long enough, close loop
                if start_node_idx in possible_next_nodes and len(current_path) >= 2: # >=2 means at least 3 nodes in loop
                    current_path.append(start_node_idx)
                    path_found = True
                    break
                
                # Else, continue path. Pick first available.
                # TODO: This choice can be problematic at junctions.
                # A better tracer would prioritize nodes with degree 2 in the remaining graph.
                next_n = -1
                for node_opt in possible_next_nodes:
                    if node_opt not in current_path[:-1]: # Avoid immediate small cycles not involving start
                        next_n = node_opt
                        break
                if next_n == -1: # If all next nodes are already in path but not start
                    if possible_next_nodes: # try any if stuck
                        next_n = possible_next_nodes[0]
                    else:
                        break # truly stuck
                
                current_path.append(next_n)
                pn = cn
                cn = next_n
            

            if path_found and len(current_path) > 3: # Valid loop: at least 3 unique vertices
                # current_path includes start_node_idx at both ends
                loop_nodes = current_path[:-1] # Remove repeating start node
                
                # Check if this loop is substantially new (not a sub-segment of existing)
                is_new_loop = True
                for existing_poly in polygons_0_based:
                    if set(loop_nodes) == set(existing_poly):
                        is_new_loop = False
                        break
                
                if is_new_loop:
                    polygons_0_based.append(loop_nodes)
                    for node_in_loop in loop_nodes:
                        globally_visited_nodes[node_in_loop] = True # Mark nodes as part of a found polygon
            # else:
                # if path_found: print(f"Debug: Path found but too short: {current_path}")
                # else: print(f"Debug: No closed path found from {start_node_idx}. Path: {current_path}")


    # Post-processing: Ensure polygons are valid (e.g., at least 3 vertices)
    valid_polygons = [p for p in polygons_0_based if len(p) >= 3]
    print(f"Identified {len(valid_polygons)} potential polygon(s).")
    return valid_polygons


def create_gmsh_geo(vertices_2d, polygons_0_based, output_geo_filepath, lc_options):
    """Creates a .geo file for Gmsh from vertices and polygon loops."""
    print(f"Generating Gmsh .geo file: {output_geo_filepath}")

    outer_polygon_idx = -1
    hole_polygon_indices = []
    
    if not polygons_0_based:
        print("Warning: No polygons provided to create_gmsh_geo. Geo file will be empty or invalid.")
        # Create an empty geo file or handle error
        with open(output_geo_filepath, 'w') as f:
            f.write("// No polygons found to mesh\n")
        return False, -1, []

    # 1. Identify outer and hole polygons based on area
    polygon_areas = []
    oriented_polygons = [] # Store polygons with consistent (CCW) orientation for outer

    for i, poly_indices in enumerate(polygons_0_based):
        poly_verts = [vertices_2d[idx] for idx in poly_indices]
        area = calculate_polygon_area(poly_verts)
        polygon_areas.append(abs(area)) # Use absolute area for sorting

        if area < 0: # Ensure CCW for consistency (Gmsh might handle it, but good practice)
            oriented_polygons.append(list(reversed(poly_indices)))
        else:
            oriented_polygons.append(list(poly_indices))

    if not polygon_areas:
        print("Error: No valid polygon areas calculated.")
        return False, -1, []

    sorted_indices_by_area = sorted(range(len(polygon_areas)), key=lambda k: polygon_areas[k], reverse=True)
    
    outer_polygon_original_idx = sorted_indices_by_area[0]
    outer_polygon_loop_0_based = oriented_polygons[outer_polygon_original_idx] # Use oriented version
    outer_poly_gmsh_tag = 1 # Gmsh outer loop tag

    # Check other polygons to see if they are holes within the outer
    hole_loops_0_based = []
    hole_gmsh_tags = []
    current_hole_tag = 2

    outer_verts_for_inpolygon = [vertices_2d[idx] for idx in outer_polygon_loop_0_based]

    for i in range(1, len(sorted_indices_by_area)):
        potential_hole_original_idx = sorted_indices_by_area[i]
        potential_hole_loop_0_based = oriented_polygons[potential_hole_original_idx]
        
        # Check if a point from the potential hole is inside the outer polygon
        # A robust check would use all points or centroid. For simplicity, use first point.
        first_pt_hole_coords = vertices_2d[potential_hole_loop_0_based[0]]
        
        # Simple point-in-polygon test (ray casting)
        # from matplotlib.path import Path # would be ideal but avoid extra dependency if possible
        # For now, assume if it's smaller and distinct, it's a candidate.
        # A proper geometric check is better. For this example, rely on area and distinctness.
        # If Gmsh gets outer {A} and inner {B,C} and B contains C, it makes B-C.
        # We only want holes directly within the main outer loop for Plane Surface (Outer, Hole1, Hole2).
        
        # A simple check: if its area is smaller and it's not identical to outer
        if polygon_areas[potential_hole_original_idx] < polygon_areas[outer_polygon_original_idx]:
            # And if it's roughly inside the bounding box of outer
            min_x_o = min(v[0] for v in outer_verts_for_inpolygon)
            max_x_o = max(v[0] for v in outer_verts_for_inpolygon)
            min_y_o = min(v[1] for v in outer_verts_for_inpolygon)
            max_y_o = max(v[1] for v in outer_verts_for_inpolygon)

            hole_candidate_verts = [vertices_2d[idx] for idx in potential_hole_loop_0_based]
            min_x_h = min(v[0] for v in hole_candidate_verts)
            max_x_h = max(v[0] for v in hole_candidate_verts) # Define max_x_h
            min_y_h = min(v[1] for v in hole_candidate_verts) # Define min_y_h
            max_y_h = max(v[1] for v in hole_candidate_verts)

            if not (min_x_h < min_x_o or max_x_h > max_x_o or \
                    min_y_h < min_y_o or max_y_h > max_y_o):
                 # crude check, but if it's not outside bounding box, consider it
                hole_loops_0_based.append(potential_hole_loop_0_based)
                hole_gmsh_tags.append(current_hole_tag)
                current_hole_tag += 1


    print(f"Outer polygon identified with {len(outer_polygon_loop_0_based)} vertices.")
    for i, hole_loop in enumerate(hole_loops_0_based):
        print(f"Hole polygon {i+1} identified with {len(hole_loop)} vertices.")

    # Use all unique vertices involved in the selected polygons for Gmsh Point definitions
    all_involved_indices_0_based = set(outer_polygon_loop_0_based)
    for hole_loop in hole_loops_0_based:
        all_involved_indices_0_based.update(hole_loop)
    
    # Create a mapping from original 0-based index to new compact 1-based Gmsh point tag
    vertex_map_orig_to_gmsh = {orig_idx: gmsh_idx + 1 
                               for gmsh_idx, orig_idx in enumerate(list(all_involved_indices_0_based))}
    
    # Determine characteristic length based on domain size
    if outer_polygon_loop_0_based:
        op_verts = [vertices_2d[i] for i in outer_polygon_loop_0_based]
        min_x = min(v[0] for v in op_verts)
        max_x = max(v[0] for v in op_verts)
        min_y = min(v[1] for v in op_verts)
        max_y = max(v[1] for v in op_verts)
        char_len = min(max_x - min_x, max_y - min_y)
        if char_len <= 0: char_len = DEFAULT_LC_GLOBAL # fallback
    else:
        char_len = DEFAULT_LC_GLOBAL

    lc_boundary_val = char_len * lc_options.get('ratio_boundary', LC_RATIO_BOUNDARY)
    lc_hole_val = char_len * lc_options.get('ratio_hole', LC_RATIO_HOLE)
    if lc_boundary_val <= 0: lc_boundary_val = DEFAULT_LC_GLOBAL / 2
    if lc_hole_val <= 0: lc_hole_val = DEFAULT_LC_GLOBAL / 4
    
    print(f"Characteristic lengths: Boundary={lc_boundary_val:.3f}, Hole={lc_hole_val:.3f}")

    with open(output_geo_filepath, 'w') as f:
        f.write("// Gmsh geometry file generated by Python script\n\n")

        # Define Points
        f.write("// Points\n")
        for orig_idx, gmsh_tag in vertex_map_orig_to_gmsh.items():
            x, y = vertices_2d[orig_idx]
            # Assign different lc based on whether point is part of hole or just outer
            lc = lc_boundary_val
            for hole_loop in hole_loops_0_based:
                if orig_idx in hole_loop:
                    lc = lc_hole_val
                    break
            f.write(f"Point({gmsh_tag}) = {{{x}, {y}, 0, {lc:.6f}}};\n")
        
        # Define Lines and Line Loops
        current_line_tag = 1
        
        # Outer loop
        f.write("\n// Outer Boundary Lines and Loop\n")
        outer_line_tags = []
        for i in range(len(outer_polygon_loop_0_based)):
            p1_orig_idx = outer_polygon_loop_0_based[i]
            p2_orig_idx = outer_polygon_loop_0_based[(i + 1) % len(outer_polygon_loop_0_based)]
            p1_gmsh_tag = vertex_map_orig_to_gmsh[p1_orig_idx]
            p2_gmsh_tag = vertex_map_orig_to_gmsh[p2_orig_idx]
            f.write(f"Line({current_line_tag}) = {{{p1_gmsh_tag}, {p2_gmsh_tag}}};\n")
            outer_line_tags.append(current_line_tag)
            current_line_tag += 1
        f.write(f"Line Loop({outer_poly_gmsh_tag}) = {{{', '.join(map(str, outer_line_tags))}}};\n")

        # Hole loops
        gmsh_hole_loop_tags = []
        for i, hole_loop_0_based in enumerate(hole_loops_0_based):
            f.write(f"\n// Hole {i+1} Lines and Loop\n")
            hole_line_tags = []
            gmsh_loop_tag_for_this_hole = hole_gmsh_tags[i] # Use the pre-assigned Gmsh tag
            gmsh_hole_loop_tags.append(gmsh_loop_tag_for_this_hole)

            for j in range(len(hole_loop_0_based)):
                p1_orig_idx = hole_loop_0_based[j]
                p2_orig_idx = hole_loop_0_based[(j + 1) % len(hole_loop_0_based)]
                p1_gmsh_tag = vertex_map_orig_to_gmsh[p1_orig_idx]
                p2_gmsh_tag = vertex_map_orig_to_gmsh[p2_orig_idx]
                f.write(f"Line({current_line_tag}) = {{{p1_gmsh_tag}, {p2_gmsh_tag}}};\n")
                hole_line_tags.append(current_line_tag)
                current_line_tag += 1
            f.write(f"Line Loop({gmsh_loop_tag_for_this_hole}) = {{{', '.join(map(str, hole_line_tags))}}};\n")

        # Define Plane Surface
        f.write("\n// Plane Surface\n")
        surface_hole_part = ""
        if gmsh_hole_loop_tags:
            surface_hole_part = f", {', '.join(map(str, gmsh_hole_loop_tags))}"
        f.write(f"Plane Surface(1) = {{{outer_poly_gmsh_tag}{surface_hole_part}}};\n")

        # Physical Groups (optional but good)
        f.write("\n// Physical Groups\n")
        f.write(f"Physical Surface(\"domain\", 101) = {{1}};\n")
        f.write(f"Physical Line(\"outer_boundary\", 201) = {{{', '.join(map(str, outer_line_tags))}}};\n")
        # Need to collect all line tags for holes for their physical group
        all_hole_line_tags_flat = []
        start_line_tag_for_holes = len(outer_line_tags) + 1
        end_line_tag_for_holes = current_line_tag -1
        if start_line_tag_for_holes <= end_line_tag_for_holes:
             all_hole_line_tags_flat = list(range(start_line_tag_for_holes, end_line_tag_for_holes +1 ))
             f.write(f"Physical Line(\"hole_boundaries\", 202) = {{{', '.join(map(str, all_hole_line_tags_flat))}}};\n")


        f.write("\n// Meshing commands\n")
        f.write("Mesh.Algorithm = 6; // Frontal-Delaunay for 2D\n")
        # f.write("Mesh.ElementOrder = 1;\n") # Linear elements
        # f.write(f"Mesh.CharacteristicLengthMax = {char_len * 0.1:.4f};\n") # Alternative global size
        f.write("Mesh 2;\n")

        output_msh_filename = os.path.splitext(output_geo_filepath)[0] + ".msh"
        f.write(f"\nSave \"{os.path.basename(output_msh_filename)}\";\n")
    
    return True, outer_poly_gmsh_tag, gmsh_hole_loop_tags


def run_gmsh(geo_filepath, gmsh_exe=GMSH_EXE_PATH):
    """Runs Gmsh to mesh the .geo file."""
    output_msh_filepath = os.path.splitext(geo_filepath)[0] + ".msh"
    print(f"Running Gmsh on {geo_filepath}...")
    command = [gmsh_exe, geo_filepath, "-2", "-o", output_msh_filepath]
    
    try:
        # Capture output for better debugging
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=120) # Timeout after 2 minutes

        if process.returncode == 0:
            print(f"Gmsh meshing successful. Output: {output_msh_filepath}")
            if stdout: print("Gmsh stdout:\n", stdout)
            if stderr: print("Gmsh stderr:\n", stderr) # Gmsh often prints info to stderr
            return True, output_msh_filepath
        else:
            print(f"Gmsh meshing failed with error code {process.returncode}.")
            print("Gmsh stdout:\n", stdout)
            print("Gmsh stderr:\n", stderr)
            return False, None
    except FileNotFoundError:
        print(f"Error: Gmsh executable not found at '{gmsh_exe}'. Please check GMSH_EXE_PATH or system PATH.")
        return False, None
    except subprocess.TimeoutExpired:
        print(f"Error: Gmsh process timed out.")
        process.kill()
        stdout, stderr = process.communicate()
        print("Gmsh stdout (on timeout):\n", stdout)
        print("Gmsh stderr (on timeout):\n", stderr)
        return False, None
    except Exception as e:
        print(f"An unexpected error occurred while running Gmsh: {e}")
        return False, None


if __name__ == "__main__":
    obj_file = "slice.obj" # Replace with your OBJ file name
    geo_file = "slice_generated.geo"
    
    if not os.path.exists(obj_file):
        print(f"Error: Input OBJ file '{obj_file}' not found.")
        # Create a dummy slice.obj for testing if it doesn't exist
        print("Creating a dummy slice.obj for testing...")
        with open(obj_file, "w") as f_dummy:
            f_dummy.write("v 0 0 0\nv 0 10 0\nv 0 10 10\nv 0 0 10\n") # Outer square
            f_dummy.write("v 0 2 2\nv 0 8 2\nv 0 8 8\nv 0 2 8\n") # Inner square
            f_dummy.write("l 1 2\nl 2 3\nl 3 4\nl 4 1\n") # Outer lines
            f_dummy.write("l 5 6\nl 6 7\nl 7 8\nl 8 5\n") # Inner lines
    
    lc_options = {
        'ratio_boundary': LC_RATIO_BOUNDARY,
        'ratio_hole': LC_RATIO_HOLE
    }

    try:
        vertices_2d, edges_1_based, _ = parse_obj(obj_file)
        
        if not vertices_2d or not edges_1_based:
            print("Error: Could not parse sufficient data from OBJ file.")
        else:
            # The polygon tracing here is a simplified version.
            # It might not be as robust as the MATLAB one for complex cases.
            # A more robust Python tracer might use a library like networkx or a more careful algorithm.
            polygons_0b = trace_polygons(len(vertices_2d), edges_1_based)

            if not polygons_0b:
                 print("Error: Polygon tracing did not identify any closed loops.")
            else:
                geo_success, _, _ = create_gmsh_geo(vertices_2d, polygons_0b, geo_file, lc_options)
                
                if geo_success:
                    run_gmsh(geo_file)
                else:
                    print("Failed to generate .geo file properly.")

    except Exception as e:
        print(f"An error occurred in the main script: {e}")
        import traceback
        traceback.print_exc()