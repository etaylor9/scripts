# -------------------------------- #
# File Info: 
    # Hexagonal Josephson Junction Array. Square chip with hexagonal lattice points 
    # disorder epsilon = 1 for superconducting regions, -0.8 for normal region
    # voltage probes
    # f = 1/3 , 2/3, 1

    # bring up an excitation at site 

    # Evloution:
    #   1. 0 --> 10 time units: thermalize
    #   2. 10 --> 110 time units: B-field applied adiabatically 
    #   3. 110 --> 600 time units: Evolve
    #   4. 600 --> 700 time units: apply excitation at site adiabatically 
    #   5. 700 --> 1000 time units: Evolve
 
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
# !!!!!  !!!!! # 

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


#  --------------------------------  Define the Functions --------------------------------  #

# Function to check if a point is inside the box around the lattice points
def is_inside_square(point, center, width, height):
    x, y = point
    x0, y0 = center
    return (abs(x - x0) <= width / 2) and (abs(y - y0) <= height / 2)

def linear_ramp(t, tmin, tmax, initial, final):
    """ Linearly interpolates the value of epsilon based on the time t. """
    if t < tmin:
        return initial
    elif t > tmax:
        return final
    else:
        return initial + (final - initial) * (t - tmin) / (tmax - tmin)
    
    

def rectangle_area(p1, p2, p3, p4, plot=False):
    """
    Calculate the area of a rectangle given four vertices and plot the rectangle.

    Parameters:
    - p1, p2, p3, p4: Tuples representing the coordinates of the four vertices.
    - flag: Boolean to indicate whether to plot the rectangle.

    Returns:
    - area: The area of the rectangle.
    """
    # Calculate the lengths of two adjacent sides
    side_length1 = distance(p1, p2)
    side_length2 = distance(p2, p3)
    
    # Check if opposite sides are equal (within a tolerance)
    side_length3 = distance(p3, p4)
    side_length4 = distance(p4, p1)
    
    if not (np.isclose(side_length1, side_length3) and np.isclose(side_length2, side_length4)):
        raise ValueError("The provided points do not form a rectangle.")

    # Area of the rectangle
    area = side_length1 * side_length2
    
    if plot:
        # Plot the rectangle on top of the original plot
        plot_rectangle(p1, p2, p3, p4)
    
    return area

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def plot_rectangle(p1, p2, p3, p4):
    """
    Plot the rectangle defined by four vertices.
    """
    x_coords = [p1[0], p2[0], p3[0], p4[0], p1[0]]
    y_coords = [p1[1], p2[1], p3[1], p4[1], p1[1]]
    plt.fill(x_coords, y_coords, 'pink', alpha=1)
    
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


#  --------------------------------  Device Geometry --------------------------------  #
# Device Geometry
total_width = 5500  # nm
total_height = 5500
width = 4100
height = 4150
width_island = 75
num_squares_per_side = 9

spacing_factor = 2

# Device Material Parameters
coherence_length = 50  # nm
london_lambda = 200  # nm
thickness = 5  # nm
gamma = 1

# Epsilon values
normal_epsilon = -0.8  # value for normal metal
epsilon0 = 1  # value for superconducting metal
excitation_epsilon = 1


step_x = width_island * spacing_factor * np.sqrt(2)
step_y = width_island * spacing_factor * np.sqrt(2)
center = (0, 0)

def is_inside_square(point, center, width, height):
    x, y = point
    x0, y0 = center
    return (abs(x - x0) <= width / 2) and (abs(y - y0) <= height / 2)

def epsilon_func(r):
    if is_inside_square(r, center, width, height):
        for r0 in r0_list:
            if np.linalg.norm(np.array(r) - np.array(r0)) <= width_island:
                return epsilon0
        return normal_epsilon
    else:
        return epsilon0

r0_list = []
for i in range(-num_squares_per_side, num_squares_per_side + 1):
    for j in range(-num_squares_per_side, num_squares_per_side + 1):
        x0 = i * step_x
        y0 = j * step_y - height/100
        if i % 2 == 1:
            y0 += step_y / 2
        point = (x0, y0)
        if is_inside_square(point, center, width, height):
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

excitation_site = (0, 2*step_x + width_island) 


# Plot epsilon distribution and circular sites
plt.figure(figsize=(10, 10))

plt.imshow(Z, extent=(-total_width / 2, total_width / 2, -total_height / 2, total_height / 2), origin='lower', cmap='coolwarm')
# plot the excitation_site as a point 

plt.scatter(*excitation_site, color='black', label='Excitation Site')

plt.colorbar(label='Epsilon value')
plt.title('Epsilon Distribution Across the Device')
plt.xlabel('X (nm)')
plt.ylabel('Y (nm)')
plt.grid(False)

# Define rectangle vertices (example)
p1 = (0, 0)
p2 = (width_island * 3 * spacing_factor, 0)
p3 = (width_island * 3* spacing_factor, width_island * spacing_factor * 1.5)
p4 = (0, width_island * spacing_factor * 1.5)

# Calculate the area and plot the rectangle
area = rectangle_area(p1, p2, p3, p4, plot = False)
print(f"The area of the rectangle is: {area:.0f} nm^2")

plot_save_path = os.path.join(file_path, f'epsilon_distribution_sym_{f:.2f}.png')
plt.savefig(plot_save_path)
plt.show()

# Build the device
layer = tdgl.Layer(london_lambda=london_lambda, coherence_length=coherence_length, thickness=thickness, gamma=gamma)
film = (
    tdgl.Polygon("film", points=box(total_width, total_height))
    .resample(1500)
    .buffer(0) #1051
)

IslandDevice = tdgl.Device(
    "JJ_arrayDevice",
    layer=layer,
    film=film,
    length_units="nm",
)

IslandDevice.make_mesh(max_edge_length=coherence_length / 2)
fig, ax = IslandDevice.plot(mesh=False, legend=True)
IslandDevice.mesh_stats()

mesh_plot_save_path = os.path.join(file_path, f'device_mesh_{f}.png')
fig.savefig(mesh_plot_save_path)


#  --------------------------------  Simulation Parameters --------------------------------  #

# calculate the number of vortices we expect to see 
h = 6.626E-34 
q = 1.602E-19
phi_0 = h/(2*q)


# Calculate the area of the rhombus formed by the points p1, p2, p3, and p4
areaUnitCell = rectangle_area(p1, p2, p3, p4, plot = False)
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
excitation_site = (0, 2*step_x + width_island) 
# !!!! CHANGE THE LOCATION OF EXCITATION SITE BEFORE YOU RUN !!!!
# Define epsilon function

def epsilon_func_time(r, *, t, excitation_site=excitation_site): 
    # excitation_site =(0, 1*width_island + 1*edge_to_edge_distance)  # Change this when you want a different excitation site

    excitation_site = excitation_site  # Define the excitation site
    t_min =  600
    t_max = 700 
    return epsilon_time(r, t, t_min = t_min, t_max = t_max, r0_list = r0_list, side_length = width_island, width = width, height = height, total_width = total_width, total_height = total_height,
                 excitation_length = step_x, excitation_site=excitation_site, epsilon0=1, normal_epsilon=normal_epsilon, excitation_epsilon= 1)


#  --------------------------------  Simulation  --------------------------------  #

options = tdgl.SolverOptions(
    solve_time= 1000,
    current_units="uA",
    field_units="mT",
    output_file=os.path.join(file_path, f"excitation_solution_sym_{f:.2f}.h5"),
)
excitation_solution = tdgl.solve(
    IslandDevice,
    options,
    applied_vector_potential = applied_vector_potential, 
    disorder_epsilon= epsilon_func_time, 
)

#  --------------------------------  Simulation Analysis --------------------------------  #

order_parameter_path = os.path.join(file_path, f'order_parameter_sym_{f:.2f}.png')
fig = excitation_solution.plot_order_parameter()
fig.savefig(order_parameter_path)

fig, ax = excitation_solution.plot_currents()
current_path = os.path.join(file_path, f'current_sym_{f:.2f}.png')
fig.savefig(current_path)

excitation_solution_video = make_video_from_solution(
    excitation_solution,
    quantities=["order_parameter", "phase", "scalar_potential"],
    figsize=(6.5, 4),
    save_dir= file_path,
)
display(excitation_solution_video)