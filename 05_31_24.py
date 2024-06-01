# initializations 
#%config InlineBackend.figure_formats = {"retina", "png"}

import os
import tempfile

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from IPython.display import HTML, display
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (5, 4)

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
    
# -------------------------------------------------------------------------------------------- #
    # Epsilon Functions

# Define the epsilon functions 
def epsilon_normal(r, r0_list, side_length, width, height, total_width=1500, total_height=1500, epsilon0=1, normal_epsilon=-0.8):
    """
    Set the disorder parameter $\epsilon$ for position r within a device.
    Epsilon = -1 : Normal Metal 
    Epsilon = +1 : Superconductor 
    """
    x, y = r
    
    # Check if r is outside the main device but within the total device area
    if abs(x) > width / 2 or abs(y) > height / 2:
        if abs(x) <= total_width / 2 and abs(y) <= total_height / 2:
            return epsilon0  # Superconducting region outside the main device area
        else:
            return normal_epsilon  # Default condition if outside the bounds of the entire device setup

    # Check if r is inside any superconducting island
    for x0, y0 in r0_list:
        if abs(x - x0) <= side_length / 2 and abs(y - y0) <= side_length / 2:
            return epsilon0  # Superconducting island

    # If not in any special region, it's normal metal
    return normal_epsilon


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

# -------------------------------------------------------------------------------------------- #
# Device Geometry 
total_width = 4000
total_height = 4000 
width = 3500
height = total_height
width_island = 280 

num_squares_per_side = 10 
spacing_factor = 1.2 # 


# Device Material Parameters 
coherence_length = 50 #nm 
london_lambda = 200 #nm 
thickness = 5 #nm
gamma = 1


# Epsilon values 
normal_epsilon = -0.8 # value for normal metal 
epsilon0 = 1 # value for superconducting metal 
excitation_epsilon = 1

# Calculate the step size for placing squares evenly
step_x = width / (num_squares_per_side + (num_squares_per_side - 1) * (spacing_factor - 1))
step_y = height / (num_squares_per_side + (num_squares_per_side - 1) * (spacing_factor - 1))

# Adjust starting point to ensure the device remains centered
start_x = -width / 2 + step_x / 2
start_y = -height / 2 + step_y / 2

# Generate r0_list
r0_list = []
for i in range(num_squares_per_side):
    for j in range(num_squares_per_side):
        x0 = start_x + i * (spacing_factor * step_x)
        y0 = start_y + j * (spacing_factor * step_y)
        r0_list.append((x0, y0))


# create a wrapper function for epsilon_normal so it makes r parameter it changes 
def epsilon_func(r):    
    return epsilon_normal(r, r0_list, side_length = width_island, width = width, height = height, total_width=total_width, total_height=total_height, epsilon0=epsilon0, normal_epsilon= normal_epsilon)

# Make the device 
length_units = "nm"

layer = tdgl.Layer(london_lambda=london_lambda, coherence_length=coherence_length, thickness=thickness, gamma = gamma)


film = (
    tdgl.Polygon("film", points=box(total_width, total_height))
    .resample(1050)
    .buffer(0) #1051
)

source = tdgl.Polygon("source", points=box(width / 100, height)).translate(dx=-total_width / 2)
drain = source.translate(dx=total_width).set_name("drain")

# Calculate step_x
step_x = width / (num_squares_per_side + (num_squares_per_side - 1) * (spacing_factor - 1))

# Calculate distance between centers
distance_between_centers = spacing_factor * step_x

# Calculate edge-to-edge distance
edge_to_edge_distance = distance_between_centers - width_island

print(f'Distance between islands: {edge_to_edge_distance:.2f} nm')
print(f'Coherence length: {coherence_length} nm')

# Make a plot of what the device will look like 
# Create a mesh grid
x = np.linspace(-total_width/2, total_width/2, 300)
y = np.linspace(-total_height/2, total_height/2, 300)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Compute epsilon for each point in the grid
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = epsilon_normal((X[j, i], Y[j, i]), r0_list, side_length = width_island, width = width, height = height, total_width= total_width, total_height = total_height)

# the probe points are ordered so that the indices 0 and 1 and the same x position but different y positions, 2 and 3 are the same x position but different y positions, and so on. The purpose of this is so that when we later calculate the voltage difference between any two probe points, we are longitudinally calculating the voltage difference
probe_points = []

# Calculate points halfway between each adjacent pair of islands on the top and bottom row
for i in range(num_squares_per_side - 1):
    x0 = start_x + i * (spacing_factor * step_x) + step_x * spacing_factor / 2  # Midpoint calculation
    y0 = start_y
    probe_points.append((x0, y0))

    y0 = start_y + (num_squares_per_side - 1) * (spacing_factor * step_y)
    probe_points.append((x0, y0))


# add the probe points to this plot: 
plt.figure(figsize=(10, 10))
plt.imshow(Z, extent=(-total_width/2, total_width/2, -total_height/2, total_height/2), origin='lower', cmap='coolwarm')
plt.colorbar(label='Epsilon value')
plt.title('Epsilon Distribution Across the Device')
plt.xlabel('X (nm)')
plt.ylabel('Y (nm)')
plt.grid(False)

plt.scatter([p[0] for p in probe_points], [p[1] for p in probe_points], marker='o', s=100, c = 'black')
plt.show()

# Build the device 
probe_points = probe_points

IslandDevice = tdgl.Device(
    "JJ_arrayDevice",
    layer=layer,
    film=film,
    terminals=[source, drain],
    length_units=length_units,
    probe_points=probe_points,
)

IslandDevice.make_mesh(max_edge_length = coherence_length/2 )

fig, ax = IslandDevice.plot(mesh=True, legend=True)

IslandDevice.mesh_stats()

# -------------------------------------------------------------------------------------------- #
# Simulate the device

# calculate the number of vortices we expect to see 
h = 6.626E-34 
q = 1.602E-19
phi_0 = h/(2*q)
applied_B = 10 #mT
areaUnitCell = (width_island*1E-9 + edge_to_edge_distance*1E-9) **2 
# number_flux = applied_B*1E-3 * total_width*1E-9 * total_height*1E-9 / phi_0 
number_flux_perUnitCell = applied_B*1E-3 * areaUnitCell / phi_0
print(f'Number of flux quanta per unit cell: {number_flux_perUnitCell:.2f}')

from tdgl.sources import LinearRamp, ConstantField
# Ramp the applied field 
applied_vector_potential = (
    LinearRamp(tmin=70, tmax=170)
    * ConstantField(applied_B, field_units="mT", length_units=IslandDevice.length_units)
)

# !!!! CHANGE THE LOCATION OF EXCITATION SITE BEFORE YOU RUN !!!!
excitation_site = (1*width_island + 1*edge_to_edge_distance, -(1*width_island + 1*edge_to_edge_distance))

# Define epsilon function
def epsilon_func_time(r, *, t, excitation_site=excitation_site): 
    # excitation_site =(0, 1*width_island + 1*edge_to_edge_distance)  # Change this when you want a different excitation site

    excitation_site = excitation_site  # Define the excitation site
    t_min =  900
    t_max = 1000 
    return epsilon_time(r, t, t_min = t_min, t_max = t_max, r0_list = r0_list, side_length = width_island, width = width, height = height, total_width = total_width, total_height = total_height,
                 excitation_length = step_x, excitation_site=excitation_site, epsilon0=1, normal_epsilon=normal_epsilon, excitation_epsilon= 1)

options = tdgl.SolverOptions(
    solve_time= 1500,
    current_units="uA",
    field_units="mT",
    output_file=os.path.join(tempdir.name, "OneBigexcitation_solution.h5"),
)
excitation_solution = tdgl.solve(
    IslandDevice,
    options,
    applied_vector_potential = applied_vector_potential, 
    disorder_epsilon= epsilon_func_time, 
)

_ = excitation_solution.plot_order_parameter()
fig, ax = excitation_solution.plot_currents()

fig, axes = excitation_solution.dynamics.plot()
excitation_solution.dynamics.plot_all_pairs()