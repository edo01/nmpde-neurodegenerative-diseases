from paraview.simple import *
import vtk
import os

import sys

if len(sys.argv) < 2:
    print("Usage: python compute_activation.py <result_dir>")
    sys.exit(1)

result = sys.argv[1]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# open .pvd file
reader = OpenDataFile(f"./results/{result}/all_output.pvd")
scene = GetAnimationScene()
times = scene.TimeKeeper.TimestepValues

# dict to hold the first‐fire time for each partition
activation = {}

for t in times:
    reader.UpdatePipeline(t)
    data = servermanager.Fetch(reader)      # vtkUnstructuredGrid
    uArr = data.GetPointData().GetArray("u")
    pArr = data.GetPointData().GetArray("partitioning")

    ncells = data.GetNumberOfCells()
    # scan points of each cell
    for cid in range(ncells):
        if cid in activation: 
            continue
        # look at all the points of that cell:
        cell = data.GetCell(cid)
        for pid in range(cell.GetNumberOfPoints()):
            ptId = cell.GetPointId(pid)
            if uArr.GetValue(ptId) >= 0.95:
                activation[cid] = t
                break

# build a new VTK array
actArr = vtk.vtkDoubleArray()
actArr.SetName("activationTime")
actArr.SetNumberOfTuples(ncells)
for cid in range(ncells):
    actArr.SetValue(cid, activation.get(cid, 0.0))

# attach it as cell data
data.GetCellData().AddArray(actArr)

# write out
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(f"./results/{result}/activation_time.vtu")
writer.SetInputData(data)
writer.Write()

print(f"Wrote {result}/activation_time.vtu")

