# -------------------------------- #
# File Info: 
    # Hexagonal Josephson Junction Array. Square chip with hexagonal lattice points 
    # disorder epsilon = 1 for superconducting regions, -0.8 for normal region
    # f = 2/3

    # Evloution:
    #   1. 0 --> 10 time units: thermalize
    #   2. 10 --> 110 time units: B-field applied adiabatically 
    #   3. 110 --> 600 time units: Evolve
 

# -------------------------------- #


# initializations 
# %config InlineBackend.figure_formats = {"retina", "png"}

import os
import tempfile

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from IPython.display import HTML, display
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (5, 4)
from tqdm import tqdm
import tdgl
import types
from tdgl.geometry import box, circle
from tdgl.visualization.animate import create_animation
MAKE_ANIMATIONS = True
tempdir = tempfile.TemporaryDirectory()

# function that makes a video of our solution 
def make_video_from_solution(
    solution,
    quantities=("order_parameter", "phase"),
    fps=20,
    figsize=(5, 4),
):
    """Generates an HTML5 video from a tdgl.Solution."""
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(
                h5file,
                quantities=quantities,
                fps=fps,
                figure_kwargs=dict(figsize=figsize),
            )
            video = anim.to_html5_video()
        return HTML(video)
    

# Function to calculate hexagon vertices
def hexagon_vertices(center, radius):
    vertices = [] 
    for i in range(6):
        angle = np.radians(60 * i)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

# Function to check if a point is inside the hexagon
def is_inside_hexagon(point, hex_vertices):
    from matplotlib.path import Path
    hex_path = Path(hex_vertices)
    return hex_path.contains_point(point)

# Function to check if a point is inside the box around the lattice points
def is_inside_square(point, center, width):
    x, y = point
    x0, y0 = center
    return (abs(x - x0) <= width / 2) and (abs(y - y0) <= width / 2)

def linear_ramp(t, tmin, tmax, initial, final):
    """ Linearly interpolates the value of epsilon based on the time t. """
    if t < tmin:
        return initial
    elif t > tmax:
        return final
    else:
        return initial + (final - initial) * (t - tmin) / (tmax - tmin)
    
    
def epsilon_time(r, t, t_min, t_max, r0_list, side_length, width, height, total_width, total_height,
                 excitation_length, excitation_site=None, epsilon0 = 1 , normal_epsilon = -0.8 , excitation_epsilon = 1 ):
    """
    Set the disorder parameter $\epsilon$ for position r within a device, with time-dependent excitations.
    Parameters: 
    epsilon0 = +1 : Superconductor 
    normal_epsilon = -0.8 : Normal Metal  
    excitation_epsilon = +1 : Superconductor 
    """
    x, y = r
    
    # Check if r is inside any superconducting island
    for x0, y0 in r0_list:
        if abs(x - x0) <= side_length / 2 and abs(y - y0) <= side_length / 2:
            return epsilon0  # Superconducting island

    # Check if r is outside the main device but within the total device area
    if abs(x) > width / 2 or abs(y) > height / 2:
        if abs(x) <= total_width / 2 and abs(y) <= total_height / 2:
            return epsilon0  # Superconducting region outside the main device area

    # Check if excitation_site is provided and within the main device area
    if excitation_site is not None and abs(x) <= width / 2 and abs(y) <= height / 2:
        x_exc, y_exc = excitation_site
        if abs(x - x_exc) <= excitation_length / 2 and abs(y - y_exc) <= excitation_length / 2:
            return linear_ramp(t, t_min, t_max, normal_epsilon, excitation_epsilon)
    
    # Default condition if not in any special region or time range
    return normal_epsilon


# Device Geometry
total_width = 4000 #nm
total_height = 4000
width = 3500
height = 3500
width_island = 75

num_squares_per_side = 12
spacing_factor = 1.8

# Device Material Parameters
coherence_length = 50  # nm
london_lambda = 200  # nm
thickness = 5  # nm
gamma = 1

# Epsilon values
normal_epsilon = -0.8  # value for normal metal
epsilon0 = 1  # value for superconducting metal
excitation_epsilon = 1

# Adjust starting point to ensure the device remains centered
start_x = -width / 2
start_y = -height / 2

step_x = width / (num_squares_per_side + (num_squares_per_side - 1) * (spacing_factor - 1))


# Define the radius of the hexagon
hex_radius = width / 2

# Calculate the vertices of the hexagon
center = (0, 0)
hex_vertices = hexagon_vertices(center, hex_radius)
hex_vertices_2 = hexagon_vertices(center, hex_radius + 20)

def epsilon_func(r):
    if is_inside_square(r, center, height):
        for r0 in r0_list:
            if np.linalg.norm(np.array(r) - np.array(r0)) <= width_island:
                return epsilon0
        return normal_epsilon
    else:
        return epsilon0
# Calculate the positions of the circular sites

r0_list = []
for i in range(-num_squares_per_side, num_squares_per_side + 1):
    for j in range(-num_squares_per_side, num_squares_per_side + 1):
        x0 = i * width_island * spacing_factor * np.sqrt(3)
        y0 = j * width_island * spacing_factor * 3 / 2
        if i % 2 == 1:
            y0 += width_island * spacing_factor * 3 / 4
        point = (x0, y0)
        if is_inside_square(point, center, width):
            r0_list.append(point)
            

# Create a higher resolution mesh grid
x = np.linspace(-total_width / 2, total_width / 2, 400)
y = np.linspace(-total_height / 2, total_height / 2, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Compute epsilon for each point in the grid
for i in tqdm(range(len(x)), desc="Computing epsilon values", unit="row"):
    for j in range(len(y)):
        Z[j, i] = epsilon_func((X[j, i], Y[j, i]))


# Plot epsilon distribution and circular sites
plt.figure(figsize=(10, 10))
plt.imshow(Z, extent=(-total_width / 2, total_width / 2, -total_height / 2, total_height / 2), origin='lower', cmap='coolwarm')
plt.colorbar(label='Epsilon value')
plt.title('Epsilon Distribution Across the Device')
plt.xlabel('X (nm)')
plt.ylabel('Y (nm)')
plt.grid(False)

# Build the device
layer = tdgl.Layer(london_lambda=london_lambda, coherence_length=coherence_length, thickness=thickness, gamma=gamma)
film = (
    tdgl.Polygon("film", points=box(total_width, total_height))
    .resample(1050)
    .buffer(0) #1051
)

IslandDevice = tdgl.Device(
    "JJ_arrayDevice",
    layer=layer,
    film=film,
    length_units="nm",
)

IslandDevice.make_mesh(max_edge_length=coherence_length / 2)
fig, ax = IslandDevice.plot(mesh=True, legend=True)
IslandDevice.mesh_stats()

# calculate the number of vortices we expect to see 
h = 6.626E-34 
q = 1.602E-19
phi_0 = h/(2*q)
f = 2/3

# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to find the area of a rhombus given four vertices
def rhombus_area(p1, p2, p3, p4):
    # Calculate the lengths of the diagonals
    diag1 = distance(p1, p3)
    diag2 = distance(p2, p4)
    # Area of the rhombus
    return 0.5 * diag1 * diag2

# Example points for the four vertices of a plaquette
p1 = (0, 0)
p2 = (spacing_factor * width_island * np.sqrt(3), 0)
p3 = (spacing_factor * width_island * np.sqrt(3) / 2, spacing_factor * width_island * 3 / 2)
p4 = (-spacing_factor * width_island * np.sqrt(3) / 2, spacing_factor * width_island * 3 / 2)

# Calculate the area of the rhombus formed by the points p1, p2, p3, and p4
areaUnitCell = rhombus_area(p1, p2, p3, p4)
print(f"The area of the plaquette (rhombus) is: {areaUnitCell:.0f} nm^2")

# number_flux_perUnitCell = applied_B*1E-3 * (areaUnitCell*1E-9*1E-9) / phi_0
print(f'Filling Factor: {f:.2f}')

applied_B = f * phi_0 * 1E3 / (areaUnitCell*1E-9*1e-9)
print(f'Applied B field is: {applied_B:.1f} mT')


from tdgl.sources import LinearRamp, ConstantField
# Ramp the applied field 
applied_vector_potential = (
    LinearRamp(tmin=10, tmax=110)
    * ConstantField(applied_B, field_units="mT", length_units=IslandDevice.length_units)
)

options = tdgl.SolverOptions(
    solve_time= 600,
    current_units="uA",
    field_units="mT",
    output_file=os.path.join(tempdir.name, "zeroExcitation_solution.h5"),
)
zeroExcitation_solution = tdgl.solve(
    IslandDevice,
    options,
    applied_vector_potential = applied_vector_potential, 
    disorder_epsilon= epsilon_func, 
)
_ = zeroExcitation_solution.plot_order_parameter()
fig, ax = zeroExcitation_solution.plot_currents()

zeroExcitation_solution_video = make_video_from_solution(
        zeroExcitation_solution,
        quantities=["order_parameter", "phase", "scalar_potential"],
        figsize=(6.5, 4),
    )
display(zeroExcitation_solution_video)