import subprocess
import os

# --- Configuration ---
GMSH_EXE_PATH = "gmsh"  # Or provide the full path if not in PATH, e.g., "/usr/local/bin/gmsh"
DEFAULT_LC_GLOBAL = 1.0  # Default characteristic length for Gmsh if not calculable
LC_RATIO_BOUNDARY = 0.03 # Target mesh size as a ratio of the characteristic length of the domain for outer boundary
LC_RATIO_HOLE = 0.015    # Finer mesh for holes
GLOBAL_MESH_SIZE_FACTOR = 0.2 # Multiplier for all characteristic lengths (e.g., 0.5 for finer, 2.0 for coarser)

def parse_obj(obj_filepath):
    """Parses an OBJ file to extract 2D vertices and line segments."""
    vertices_3d = []
    vertices_2d = []
    edges = []

    print(f"Parsing OBJ file: {obj_filepath}")
    with open(obj_filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            
            cmd = parts[0]
            if cmd == 'v':
                try:
                    y = float(parts[2])
                    z = float(parts[3])
                    vertices_3d.append((float(parts[1]), y, z))
                    vertices_2d.append((y, z))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed vertex line {line_num}: {line.strip()} - {e}")
            elif cmd == 'l':
                try:
                    v1_idx = int(parts[1])
                    v2_idx = int(parts[2])
                    edges.append((v1_idx, v2_idx))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line segment line {line_num}: {line.strip()} - {e}")
    
    print(f"Found {len(vertices_2d)} vertices and {len(edges)} edges.")
    if not vertices_2d:
        raise ValueError("No vertices found in OBJ file.")
    if not edges:
        print("Warning: No line segments ('l' commands) found in OBJ file. Cannot form loops.")
    return vertices_2d, edges, vertices_3d

def calculate_polygon_area(polygon_vertices):
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
    print("Identifying contours (tracing polygons)...")
    adj = [[] for _ in range(num_vertices)]
    for v1_obj, v2_obj in edges_1_based:
        u, v = v1_obj - 1, v2_obj - 1
        if 0 <= u < num_vertices and 0 <= v < num_vertices:
            adj[u].append(v)
            adj[v].append(u)
        else:
            print(f"Warning: Edge ({v1_obj}, {v2_obj}) references out-of-bounds vertex index. Max index: {num_vertices-1}. Skipping edge.")

    for i in range(num_vertices):
        adj[i] = list(set(adj[i]))

    polygons_0_based = []
    globally_visited_nodes = [False] * num_vertices

    for start_node_idx in range(num_vertices):
        if not globally_visited_nodes[start_node_idx] and adj[start_node_idx]:
            current_path = [start_node_idx]
            cn = start_node_idx
            pn = -1 
            path_found = False

            while len(current_path) <= num_vertices : 
                possible_next_nodes = []
                for neighbor in adj[cn]:
                    if neighbor == pn:
                        continue
                    possible_next_nodes.append(neighbor)
                
                if not possible_next_nodes:
                    break 

                if start_node_idx in possible_next_nodes and len(current_path) >= 2: 
                    current_path.append(start_node_idx)
                    path_found = True
                    break
                
                next_n = -1
                # Try to pick a node not yet in path (excluding the immediate previous, which is already handled)
                for node_opt in possible_next_nodes:
                    if node_opt not in current_path: # Check against entire current_path
                        next_n = node_opt
                        break
                
                if next_n == -1: # All candidates are already in path (but not start_node_idx yet)
                                 # or no candidates other than going back.
                                 # This implies a more complex situation or a dead-end for simple loop.
                                 # For simple non-intersecting loops, this shouldn't be hit often if logic is right.
                                 # If still no next_n, try any from possible_next_nodes if it's not start_node_idx
                                 # and not already the vast majority of the path (to avoid tiny loops).
                    for node_opt in possible_next_nodes:
                         if node_opt != start_node_idx : # Avoid premature closing if not truly the end
                            next_n = node_opt # Fallback: pick first available if stuck
                            break
                    if next_n == -1 and possible_next_nodes: # If only option is start_node but path not long enough
                         break # Avoid getting stuck in tiny loop not forming a polygon

                if next_n == -1: # Truly stuck
                    break
                
                current_path.append(next_n)
                pn = cn
                cn = next_n
            
            if path_found and len(current_path) > 3: # Path: [s, n1, n2, ..., s]. Loop: [s, n1, n2, ...]
                loop_nodes = current_path[:-1] 
                
                is_new_loop = True
                # Check against set of nodes to avoid adding permutations of the same loop
                set_loop_nodes = set(loop_nodes)
                for existing_poly in polygons_0_based:
                    if set(existing_poly) == set_loop_nodes:
                        is_new_loop = False
                        break
                
                if is_new_loop:
                    polygons_0_based.append(loop_nodes)
                    for node_in_loop in loop_nodes:
                        globally_visited_nodes[node_in_loop] = True 
    
    valid_polygons = [p for p in polygons_0_based if len(p) >= 3]
    print(f"Identified {len(valid_polygons)} potential polygon(s):")
    for i, p in enumerate(valid_polygons):
        print(f"  Polygon {i}: {len(p)} vertices, indices (0-based): {p[:5]}...{p[-2:] if len(p)>5 else ''}")
    return valid_polygons


def create_gmsh_geo(vertices_2d, polygons_0_based, output_geo_filepath, lc_options):
    print(f"Generating Gmsh .geo file: {output_geo_filepath}")

    if not polygons_0_based:
        print("Warning: No polygons provided to create_gmsh_geo. Geo file will be empty or invalid.")
        with open(output_geo_filepath, 'w') as f:
            f.write("// No polygons found to mesh\n")
        return False, -1, []

    polygon_areas_signed = []
    oriented_polygons = [] 

    for i, poly_indices in enumerate(polygons_0_based):
        poly_verts = [vertices_2d[idx] for idx in poly_indices]
        area = calculate_polygon_area(poly_verts)
        polygon_areas_signed.append(area)

        # Gmsh generally prefers CCW for outer, CW for inner for Plane Surface definition
        # Let's make outer CCW and inner CW by convention here.
        # We'll use absolute area for size comparison later.
        if i == 0: # Assume first found (largest area after sorting) is outer. Make it CCW.
            if area < 0:
                oriented_polygons.append(list(reversed(poly_indices)))
                polygon_areas_signed[i] *= -1 # store positive area
            else:
                oriented_polygons.append(list(poly_indices))
        else: # For potential holes, make them CW
            if area > 0:
                oriented_polygons.append(list(reversed(poly_indices)))
                polygon_areas_signed[i] *= -1 # store negative area
            else:
                oriented_polygons.append(list(poly_indices))


    # Sort by absolute area to find the largest (outermost)
    # Store original index to retrieve oriented polygon later
    # We need to sort by *absolute* area to find the outer boundary reliably
    # Then, when constructing Plane Surface, Gmsh expects outer loop CCW, inner loops CW.
    
    # Re-evaluate orientation after sorting
    # 1. Find largest polygon by absolute area - this is the outer.
    # 2. Ensure it's CCW.
    # 3. For all others (potential holes), ensure they are CW.

    abs_areas = [abs(calculate_polygon_area([vertices_2d[idx] for idx in p])) for p in polygons_0_based]
    if not abs_areas:
        print("Error: No valid polygon areas calculated (all polygons have < 3 vertices).")
        return False, -1, []

    sorted_original_indices_by_abs_area = sorted(range(len(polygons_0_based)), key=lambda k: abs_areas[k], reverse=True)
    
    # Outer polygon
    outer_polygon_original_idx = sorted_original_indices_by_abs_area[0]
    outer_polygon_loop_0_based_temp = polygons_0_based[outer_polygon_original_idx]
    outer_area_signed = calculate_polygon_area([vertices_2d[idx] for idx in outer_polygon_loop_0_based_temp])
    if outer_area_signed < 0: # Ensure CCW
        outer_polygon_loop_0_based = list(reversed(outer_polygon_loop_0_based_temp))
    else:
        outer_polygon_loop_0_based = list(outer_polygon_loop_0_based_temp)
    outer_poly_gmsh_tag = 1

    hole_loops_0_based = []
    hole_gmsh_tags = []
    current_hole_tag = 2
    outer_verts_for_inpolygon = [vertices_2d[idx] for idx in outer_polygon_loop_0_based]

    for i in range(1, len(sorted_original_indices_by_abs_area)):
        potential_hole_original_idx = sorted_original_indices_by_abs_area[i]
        potential_hole_loop_0_based_temp = polygons_0_based[potential_hole_original_idx]
        
        # Crude check: if smaller area and BBox inside outer's BBox.
        # A proper point-in-polygon for centroid would be more robust.
        min_x_o = min(v[0] for v in outer_verts_for_inpolygon)
        max_x_o = max(v[0] for v in outer_verts_for_inpolygon)
        min_y_o = min(v[1] for v in outer_verts_for_inpolygon)
        max_y_o = max(v[1] for v in outer_verts_for_inpolygon)

        hole_candidate_verts = [vertices_2d[idx] for idx in potential_hole_loop_0_based_temp]
        min_x_h = min(v[0] for v in hole_candidate_verts)
        max_x_h = max(v[0] for v in hole_candidate_verts)
        min_y_h = min(v[1] for v in hole_candidate_verts)
        max_y_h = max(v[1] for v in hole_candidate_verts)

        if not (min_x_h < min_x_o or max_x_h > max_x_o or \
                min_y_h < min_y_o or max_y_h > max_y_o):
            # Potential hole is inside bounding box of outer.
            # Ensure it's CW for Gmsh Plane Surface definition
            hole_area_signed = calculate_polygon_area(hole_candidate_verts)
            if hole_area_signed > 0: # If CCW, reverse to make it CW
                potential_hole_loop_0_based = list(reversed(potential_hole_loop_0_based_temp))
            else:
                potential_hole_loop_0_based = list(potential_hole_loop_0_based_temp)
            
            hole_loops_0_based.append(potential_hole_loop_0_based)
            hole_gmsh_tags.append(current_hole_tag)
            current_hole_tag += 1

    print(f"Outer polygon (tag {outer_poly_gmsh_tag}) identified with {len(outer_polygon_loop_0_based)} vertices.")
    for i, hole_loop in enumerate(hole_loops_0_based):
        print(f"Hole polygon (tag {hole_gmsh_tags[i]}) identified with {len(hole_loop)} vertices.")

    all_involved_indices_0_based = set(outer_polygon_loop_0_based)
    for hole_loop in hole_loops_0_based:
        all_involved_indices_0_based.update(hole_loop)
    
    vertex_map_orig_to_gmsh = {orig_idx: gmsh_idx + 1 
                               for gmsh_idx, orig_idx in enumerate(list(all_involved_indices_0_based))}
    
    char_len = DEFAULT_LC_GLOBAL
    if outer_polygon_loop_0_based:
        op_verts = [vertices_2d[i] for i in outer_polygon_loop_0_based]
        min_x = min(v[0] for v in op_verts)
        max_x = max(v[0] for v in op_verts)
        min_y = min(v[1] for v in op_verts)
        max_y = max(v[1] for v in op_verts)
        # Use average of width and height for a more general char_len
        dx = max_x - min_x
        dy = max_y - min_y
        if dx > 1e-9 and dy > 1e-9: # Ensure it's a 2D extent
             char_len = (dx + dy) / 2.0
        elif dx > 1e-9:
             char_len = dx
        elif dy > 1e-9:
             char_len = dy
        else: # Fallback if very small or degenerate
             print(f"Warning: Outer polygon has minimal extent (dx={dx}, dy={dy}). Using default char_len.")
             char_len = DEFAULT_LC_GLOBAL
    print(f"Calculated characteristic length of domain (char_len): {char_len:.4f}")


    lc_boundary_val = char_len * lc_options.get('ratio_boundary', LC_RATIO_BOUNDARY)
    lc_hole_val = char_len * lc_options.get('ratio_hole', LC_RATIO_HOLE)
    global_factor = lc_options.get('global_factor', 1.0)

    if lc_boundary_val <= 1e-9: lc_boundary_val = DEFAULT_LC_GLOBAL / 2.0
    if lc_hole_val <= 1e-9: lc_hole_val = DEFAULT_LC_GLOBAL / 4.0
    
    # Apply global factor AFTER specific ratios are calculated
    # lc_boundary_val *= global_factor # No, CharacteristicLengthFactor does this
    # lc_hole_val *= global_factor     # No, CharacteristicLengthFactor does this

    print(f"Target characteristic lengths (before global factor): Boundary={lc_boundary_val:.4f}, Hole={lc_hole_val:.4f}")
    print(f"Global mesh size factor to be applied by Gmsh: {global_factor:.3f}")


    with open(output_geo_filepath, 'w') as f:
        f.write("// Gmsh geometry file generated by Python script\n\n")
        
        # Apply global mesh size factor
        f.write(f"Mesh.CharacteristicLengthFactor = {global_factor:.6f};\n\n")

        f.write("// Points\n")
        for orig_idx, gmsh_tag in vertex_map_orig_to_gmsh.items():
            x, y = vertices_2d[orig_idx]
            lc = lc_boundary_val # Default for outer boundary points
            # Check if this point belongs to any identified hole loop
            is_hole_point = False
            for hole_loop in hole_loops_0_based:
                if orig_idx in hole_loop:
                    lc = lc_hole_val
                    is_hole_point = True
                    break
            f.write(f"Point({gmsh_tag}) = {{{x}, {y}, 0, {lc:.6f}}};\n") # lc here is BEFORE global factor
        
        current_line_tag = 1
        
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

        gmsh_hole_loop_tags = []
        all_hole_line_tags_for_physical_group = []
        for i, hole_loop_0_based in enumerate(hole_loops_0_based):
            f.write(f"\n// Hole {i+1} Lines and Loop\n")
            hole_line_tags_this_loop = []
            gmsh_loop_tag_for_this_hole = hole_gmsh_tags[i]
            gmsh_hole_loop_tags.append(gmsh_loop_tag_for_this_hole)

            for j in range(len(hole_loop_0_based)):
                p1_orig_idx = hole_loop_0_based[j]
                p2_orig_idx = hole_loop_0_based[(j + 1) % len(hole_loop_0_based)]
                p1_gmsh_tag = vertex_map_orig_to_gmsh[p1_orig_idx]
                p2_gmsh_tag = vertex_map_orig_to_gmsh[p2_orig_idx]
                f.write(f"Line({current_line_tag}) = {{{p1_gmsh_tag}, {p2_gmsh_tag}}};\n")
                hole_line_tags_this_loop.append(current_line_tag)
                all_hole_line_tags_for_physical_group.append(current_line_tag)
                current_line_tag += 1
            f.write(f"Line Loop({gmsh_loop_tag_for_this_hole}) = {{{', '.join(map(str, hole_line_tags_this_loop))}}};\n")

        f.write("\n// Plane Surface\n")
        surface_hole_part = ""
        if gmsh_hole_loop_tags:
            surface_hole_part = f", {', '.join(map(str, gmsh_hole_loop_tags))}"
        # Surface tag 1 for the main domain
        plane_surface_tag = 1 
        f.write(f"Plane Surface({plane_surface_tag}) = {{{outer_poly_gmsh_tag}{surface_hole_part}}};\n")

        f.write("\n// Physical Groups\n")
        f.write(f"Physical Surface(\"domain\", 101) = {{{plane_surface_tag}}};\n")
        if outer_line_tags:
            f.write(f"Physical Line(\"outer_boundary\", 201) = {{{', '.join(map(str, outer_line_tags))}}};\n")
        if all_hole_line_tags_for_physical_group:
             f.write(f"Physical Line(\"hole_boundaries\", 202) = {{{', '.join(map(str, all_hole_line_tags_for_physical_group))}}};\n")

        f.write("\n// Meshing commands\n")
        f.write("Mesh.Algorithm = 6; // Frontal-Delaunay for 2D ( alternatives: 5 for Delaunay, 8 for Automatic)\n")
        # f.write("Mesh.ElementOrder = 1;\n") 
        f.write("Mesh 2;\n")

        output_msh_filename = os.path.splitext(os.path.basename(output_geo_filepath))[0] + ".msh"
        f.write(f"\nSave \"{output_msh_filename}\";\n") # Use relative path for Save
    
    return True, outer_poly_gmsh_tag, gmsh_hole_loop_tags


def run_gmsh(geo_filepath, gmsh_exe=GMSH_EXE_PATH):
    output_msh_filepath = os.path.splitext(geo_filepath)[0] + ".msh"
    # Ensure output_msh_filepath in Popen is just the filename if geo_filepath is just filename
    # Or make it absolute if geo_filepath is absolute. Gmsh will create it in CWD if path is relative.
    # The "Save" command inside the .geo file now uses a relative path, so Gmsh will save it
    # next to the .geo file (or in its CWD if that's different).
    # The -o flag in the command line overrides the Save command in the script.
    
    # Let's ensure the -o path is robust:
    output_msh_abs_filepath = os.path.abspath(output_msh_filepath)

    print(f"Running Gmsh on {geo_filepath}...")
    # Command: gmsh input.geo -2 -o output.msh
    # The -o output_msh_filepath will override the Save command in the .geo file.
    command = [gmsh_exe, geo_filepath, "-2", "-o", output_msh_abs_filepath, "-format", "msh2"] # Added msh2 format
    
    try:
        print(f"Executing command: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=os.path.dirname(os.path.abspath(geo_filepath)))
        stdout, stderr = process.communicate(timeout=120)

        if process.returncode == 0:
            print(f"Gmsh meshing successful. Output: {output_msh_abs_filepath}")
            if stdout: print("Gmsh stdout:\n", stdout)
            # Gmsh often prints info to stderr even on success
            if stderr: print("Gmsh stderr (Info/Warnings):\n", stderr)
            return True, output_msh_abs_filepath
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
    obj_file = "slice.obj" 
    base_name = os.path.splitext(obj_file)[0]
    geo_file = f"{base_name}_generated.geo" # Use a more descriptive name
    
    if not os.path.exists(obj_file):
        print(f"Error: Input OBJ file '{obj_file}' not found.")
        print("Creating a dummy slice.obj for testing...")
        with open(obj_file, "w") as f_dummy:
            f_dummy.write("# Dummy OBJ File\n")
            f_dummy.write("# Outer square (CCW from top-left in y-z plane -> x_2d, y_2d)\n")
            f_dummy.write("v 0 0 10\n")  # v1 (0,10)
            f_dummy.write("v 0 10 10\n") # v2 (10,10)
            f_dummy.write("v 0 10 0\n")  # v3 (10,0)
            f_dummy.write("v 0 0 0\n")   # v4 (0,0)
            f_dummy.write("# Inner square (CW from top-left in y-z plane -> x_2d, y_2d for Gmsh hole)\n")
            f_dummy.write("v 0 2 8\n")   # v5 (2,8)
            f_dummy.write("v 0 2 2\n")   # v6 (2,2)
            f_dummy.write("v 0 8 2\n")   # v7 (8,2)
            f_dummy.write("v 0 8 8\n")   # v8 (8,8)
            f_dummy.write("# Outer lines\n")
            f_dummy.write("l 1 2\nl 2 3\nl 3 4\nl 4 1\n")
            f_dummy.write("# Inner lines\n")
            f_dummy.write("l 5 6\nl 6 7\nl 7 8\nl 8 5\n")
    
    # --- TRY DIFFERENT SETTINGS HERE ---
    lc_options = {
        'ratio_boundary': LC_RATIO_BOUNDARY, # e.g., 0.05 for coarser, 0.01 for finer
        'ratio_hole': LC_RATIO_HOLE,         # e.g., 0.025 for coarser, 0.005 for finer
        'global_factor': GLOBAL_MESH_SIZE_FACTOR # e.g., 0.5 for finer, 2.0 for coarser overall
    }
    # Example: For a much finer mesh:
    # lc_options['ratio_boundary'] = 0.01
    # lc_options['ratio_hole'] = 0.005
    # lc_options['global_factor'] = 0.5

    # Example: For a coarser mesh:
    # lc_options['ratio_boundary'] = 0.1
    # lc_options['ratio_hole'] = 0.05
    # lc_options['global_factor'] = 1.5


    try:
        vertices_2d, edges_1_based, _ = parse_obj(obj_file)
        
        if not vertices_2d or not edges_1_based:
            print("Error: Could not parse sufficient data from OBJ file.")
        else:
            polygons_0b = trace_polygons(len(vertices_2d), edges_1_based)

            if not polygons_0b:
                 print("Error: Polygon tracing did not identify any closed loops. Cannot proceed.")
            else:
                geo_success, _, _ = create_gmsh_geo(vertices_2d, polygons_0b, geo_file, lc_options)
                
                if geo_success:
                    print(f"\n--- Inspect '{geo_file}' manually to verify Point lc values and CharacteristicLengthFactor ---\n")
                    success, msh_file = run_gmsh(geo_file)
                    if success:
                        print(f"Successfully generated mesh: {msh_file}")
                        # You can add code here to parse the .msh file and count elements if needed
                    else:
                        print("Gmsh execution failed.")
                else:
                    print("Failed to generate .geo file properly.")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}")
        import traceback
        traceback.print_exc()