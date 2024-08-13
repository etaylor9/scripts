# -------------------------------- #
# File Info: August 12th, 2024
    # Goals for this file: 
    #     - first round of kagome lattice 
# -------------------------------- #

import os
import tempfile
import argparse
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from IPython.display import HTML, display
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tdgl
import types
from tdgl.geometry import box, circle
from tdgl.visualization.animate import create_animation

MAKE_ANIMATIONS = True
tempdir = tempfile.TemporaryDirectory()

parser = argparse.ArgumentParser(description='Hexagonal Josephson Junction Array Simulation')
parser.add_argument('--f', type=float, required=True, help='Filling factor')
parser.add_argument('--outdir', type=str, required=True, help='Output directory')

args = parser.parse_args()

f = args.f
file_path = args.outdir
print(f'Simulating a Kagome lattice for a filling factor f = {f:.2f} ')

# !!!!!  !!!!! # 

#  --------------------------------  Define the Functions --------------------------------  #
def make_video_from_solution(
    solution,
    quantities=("order_parameter", "phase"),
    fps=20,
    figsize=(5, 4),
    save_dir= file_path,
):
    """Generates an HTML5 video from a tdgl.Solution and saves it to a file."""
    save_path = os.path.join(save_dir, f'solution_video_{f:.2f}.mp4')
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(
                h5file,
                quantities=quantities,
                fps=fps,
                figure_kwargs=dict(figsize=figsize),
            )
            # Save the animation to the specified file
            anim.save(save_path, writer='ffmpeg')
            video = anim.to_html5_video()
        return HTML(video)
    

# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to find the area of a rhombus given four vertices
def rhombus_area(p1, p2, p3, p4): #go around the rhombus in a clockwise direction
    # Calculate the lengths of the diagonals
    diag1 = distance(p1, p3)
    diag2 = distance(p2, p4)
    # Area of the rhombus
    return 0.5 * diag1 * diag2

# Define the function to get epsilon value based on position
def get_epsilon(r, connections, thickness):
    x, y = r 
    for (p1, p2) in connections:
        # Check if p1 and p2 are the same to avoid division by zero
        if np.array_equal(p1, p2):
            continue  # Skip this connection if p1 == p2
        
        # Calculate the distance from point (x, y) to the line segment p1-p2
        norm_p2_p1 = np.linalg.norm(p2 - p1)
        if norm_p2_p1 == 0:
            continue  # Skip if the line segment length is zero
        
        d = np.abs(np.cross(p2-p1, p1-np.array([x, y]))) / norm_p2_p1
        
        if d <= thickness:
            return epsilon_superconducting
    return epsilon_normal

def generate_kagome_lattice(total_width, total_height, a, shift_x=0, shift_y=0):
    # Extend the grid size to ensure full coverage
    extended_width = total_width + 4 * a * np.sqrt(3)
    extended_height = total_height + 4 * a * np.sqrt(3)

    num_cells_x = int(np.ceil(extended_width / (a * 1.5))) + 6
    num_cells_y = int(np.ceil(extended_height / (a * np.sqrt(3)))) + 6

    pos = np.array([[0, 0], [a, 0], [0.5 * a, 0.5 * a * np.sqrt(3)]])
    points = []
    connections = []

    # Apply the user-defined shift to the base position
    shift = np.array([shift_x, shift_y])

    for i in tqdm(range(num_cells_x), desc="Creating centered kagome lattice with adjustable shift.."):
        for j in range(num_cells_y):
            base = i * pos[1] * 2 + j * pos[2] * 2 + shift
            cell_points = [base + p for p in pos]
            points.extend(cell_points)

            # Connections within the unit cell
            connections.append((cell_points[0], cell_points[1]))
            connections.append((cell_points[0], cell_points[2]))
            connections.append((cell_points[1], cell_points[2]))

            # Horizontal connection to the next cell to the right
            if i < num_cells_x - 1:
                right_base = base + pos[1] * 2
                right_point = right_base + pos[0]
                connections.append((cell_points[1], right_point))

            # Vertical connection to the cell above
            if j < num_cells_y - 1:
                above_base = base + pos[2] * 2
                above_point = above_base + pos[0]
                connections.append((cell_points[2], above_point))

    points = np.array(points)

    # Define bounds for the centered region
    min_x = -total_width / 2
    max_x = total_width / 2
    min_y = -total_height / 2
    max_y = total_height / 2

    # Filter points and connections to only include those within the centered region
    valid_points_mask = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    filtered_points = points[valid_points_mask]

    filtered_connections = [(p1, p2) for p1, p2 in connections 
                            if (p1[0] >= min_x and p1[0] <= max_x and p1[1] >= min_y and p1[1] <= max_y and 
                                p2[0] >= min_x and p2[0] <= max_x and p2[1] >= min_y and p2[1] <= max_y)]

    return filtered_points, filtered_connections


def get_epsilon_wrapped(r):    
    return get_epsilon(r, connections, thickness_connection)

#  --------------------------------  Device Geometry --------------------------------  #
# Device Geometry
total_width = 8E3 # nm
total_height = 8E3 # nm 

# Device Material Parameters
thickness_film = 5 # nm
thickness_connection = 150 # nm
gamma = 1

# Epsilon values
epsilon_normal = -1 # value for normal metal
epsilon_superconducting = 1 # value for superconducting metal
excitation_epsilon = 1
london_lambda = 500 # nm 
coherence_length = 80 # nm 
a = 680 # Lattice constant


#  --------------------------------  Create the Kagome Lattice --------------------------------  #

#  --------------------------------  Create the Kagome Lattice --------------------------------  #
# Allow manual adjustment of the shift
shift_x = -11000 # Start with no shift in x
shift_y = -6000  # Start with no shift in y

points, connections = generate_kagome_lattice(total_width, total_height, a, shift_x, shift_y)

plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], s=10)
plt.xlabel("X [nm]")
plt.ylabel("Y [nm]")

# Define rectangle vertices (example)
p1 = (points[23][0], points[23][1])
p2 = (points[26][0], points[26][1])
p3 = (points[46][0], points[46][1]) 
p4 =  (points[43][0], points[43][1])


plt.scatter(p1[0], p1[1],  color = "red")
plt.scatter(p2[0], p2[1], color = "green")
plt.scatter(p3[0], p3[1], color = "orange")
plt.scatter(p4[0], p4[1], color = "purple")


plt.gca().set_aspect('equal', adjustable='box')
plot_save_path = os.path.join(file_path, f'kagome_lattice.png')
plt.savefig(plot_save_path)
plt.show()

#  --------------------------------  Create the Epsilon Distribution --------------------------------  #
# Define a grid to visualize epsilon values
grid_res = 250  # Resolution of the grid
x_grid = np.linspace(-total_width/2, total_width/2, grid_res)
y_grid = np.linspace(-total_width/2, total_width/2, grid_res)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.zeros_like(X)

# Calculate epsilon values for each point on the grid with a progress bar
for i in tqdm(range(grid_res), desc="Calculating epsilon values"):
    for j in range(grid_res):
        Z[i, j] = get_epsilon((X[i, j], Y[i, j]), connections, thickness_connection)

# Plot the epsilon distribution
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=1, cmap='coolwarm')
plt.colorbar(label='Epsilon Value')

plt.title('Epsilon Distribution on Kagome Lattice')
plt.xlabel("X [nm]")
plt.ylabel("Y [nm]")

plot_save_path = os.path.join(file_path, f'epsilon_distribution.png')
plt.savefig(plot_save_path)
plt.show()


#  --------------------------------  Build the Device --------------------------------  #
# Make the device 
length_units = "nm"
layer = tdgl.Layer(london_lambda=london_lambda, coherence_length=coherence_length, thickness=thickness_film, gamma = gamma)

film = (
    tdgl.Polygon("film", points=box(total_width, total_height))
    .resample(1500)
    .buffer(0)
)

# Build the device
IslandDevice = tdgl.Device(
    "JJ_arrayDevice",
    layer=layer,
    film=film,
    length_units=length_units,
)

IslandDevice.make_mesh(max_edge_length = coherence_length/2)
fig, ax = IslandDevice.plot(mesh=True, legend=True)
IslandDevice.mesh_stats()



#  --------------------------------  Simulation Parameters --------------------------------  #
# calculate the number of vortices we expect to see 
h = 6.626E-34 
q = 1.602E-19
phi_0 = h/(2*q)

# make sure the values go around in a clockwise fashion 
areaUnitCell = rhombus_area(p1, p2, p3, p4)
print(f"The area of the plaquette (rhombus) is: {areaUnitCell:.0f} nm^2")

print(f'Filling Factor: {f:.2f}')

factor = 1 
applied_B = f * phi_0 * 1E3 / (areaUnitCell*1E-9*1e-9)  / factor  
print(f'Applied B field is: {applied_B:.3f} mT for a reduced factor of {factor}')


from tdgl.sources import LinearRamp, ConstantField
# Ramp the applied field 
applied_vector_potential = (
    LinearRamp(tmin=10, tmax=110)
    * ConstantField(applied_B, field_units="mT", length_units=IslandDevice.length_units)
)

#  --------------------------------  Run Simulation --------------------------------  #
options = tdgl.SolverOptions(
    solve_time= 600,
    current_units="uA",
    field_units="mT",
    output_file=os.path.join(tempdir.name, "praying.h5"),
)
solution = tdgl.solve(
    IslandDevice,
    options,
    applied_vector_potential = applied_vector_potential, 
    disorder_epsilon= get_epsilon_wrapped, 
)

#  --------------------------------  Simulation Analysis --------------------------------  #

order_parameter_path = os.path.join(file_path, f'order_parameter_{f:.2f}.png')
fig , ax = solution.plot_order_parameter()
fig.savefig(order_parameter_path)

fig, ax = solution.plot_currents()
current_path = os.path.join(file_path, f'current_{f:.2f}.png')
fig.savefig(current_path)

solution_video = make_video_from_solution(
    solution,
    quantities=["order_parameter", "phase", "scalar_potential"],
    figsize=(6.5, 4),
    save_dir= file_path,
)
display(solution_video)