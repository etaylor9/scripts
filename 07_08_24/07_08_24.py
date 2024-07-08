# -------------------------------- #
# File Info: July 8th, 2024
    # Goals for this file: 
    #     - reduce the B-field by a factor of 5 instead of 10 to hopefully get closer to the correct filling factor
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


def plot_rectangle(p1, p2, p3, p4):
    """
    Plot the rectangle defined by four vertices.
    """
    x_coords = [p1[0], p2[0], p3[0], p4[0], p1[0]]
    y_coords = [p1[1], p2[1], p3[1], p4[1], p1[1]]
    plt.fill(x_coords, y_coords, 'pink', alpha=1)


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

#  --------------------------------  Device Geometry --------------------------------  #
# Device Geometry
total_width = 30E3  # nm
total_height = 30E3
width = 25E3
height = 26E3
width_island = 1E3  # nm
num_squares_per_side = 20

spacing_factor = 1.2

# Device Material Parameters
coherence_length = 600  # nm
london_lambda = 16  # nm
thickness = 5  # nm
gamma = 1

# Epsilon values
normal_epsilon = -0.8  # value for normal metal
epsilon0 = 1  # value for superconducting metal
excitation_epsilon = 1

step_x = width_island * spacing_factor * np.sqrt(1)
step_y = width_island * spacing_factor * np.sqrt(1)

center = (0, 0)

#  --------------------------------  Compute the Epsilon Distribution --------------------------------  #
def epsilon_func(r):
    if is_inside_square(r, center, width, height):
        for r0 in r0_list:
            if (np.linalg.norm(np.array(r) - np.array(r0)) <= width_island/2):
                # Check if the entire site is within the array
                x0, y0 = r0
                if (abs(x0 - center[0]) <= (width - width_island) / 2 and 
                    abs(y0 - center[1]) <= (height - width_island) / 2):
                    return epsilon0
        return normal_epsilon
    else:
        return epsilon0
    
# compute the lattice points
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
x = np.linspace(-total_width / 2, total_width / 2, 500)
y = np.linspace(-total_height / 2, total_height / 2, 500)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Compute epsilon for each point in the grid
for i in tqdm(range(len(x)), desc="Computing epsilon values", unit="row"):
    for j in range(len(y)):
        Z[j, i] = epsilon_func((X[j, i], Y[j, i]))


# -------------------------------- Plot the Epsilon Distribution --------------------------------  #
# Plot epsilon distribution and circular sites
plt.figure(figsize=(10, 10))

plt.imshow(Z, extent=(-total_width / 2, total_width / 2, -total_height / 2, total_height / 2), origin='lower', cmap='coolwarm')
# plot the excitation_site as a point 

plt.colorbar(label='Epsilon value')
plt.title('Epsilon Distribution Across the Device')
plt.xlabel('X (nm)')
plt.ylabel('Y (nm)')
plt.grid(False)

# Define rectangle vertices (example)
p1 = (r0_list[0][0], r0_list[0][1])
p2 = (r0_list[1][0], r0_list[1][1])
p3 = (r0_list[24][0], r0_list[24][1]) 
p4 =  (r0_list[23][0], r0_list[23][1]) 


# make sure the values go around in a clockwise fashion 
areaUnitCell = rhombus_area(p1, p2, p3, p4)
print(f"Plaquette Area (rhombus): {areaUnitCell:.0f} nm^2")

dist_btw_island = distance(p1,p2)
print(f"Island Size: {width_island:.0f} nm")
print(f"Center to center distance between islands: {dist_btw_island:.0f} nm")
print(f"Edge to Edge distance between islands: {dist_btw_island - width_island:.0f} nm")

plt.scatter(p1[0], p1[1], color='black')
plt.scatter(p2[0], p2[1], color='black')
plt.scatter(p3[0], p3[1], color='black')
plt.scatter(p4[0], p4[1], color='black')

plot_save_path = os.path.join(file_path, f'epsilon_distribution_sym_{f:.2f}.png')
plt.savefig(plot_save_path)
plt.show()

#  --------------------------------  Device Construction --------------------------------  #
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

mesh_plot_save_path = os.path.join(file_path, f'device_mesh_{f:.2f}.png')
fig.savefig(mesh_plot_save_path)


#  --------------------------------  Simulation Parameters --------------------------------  #
# calculate the number of vortices we expect to see 
h = 6.626E-34 
q = 1.602E-19
phi_0 = h/(2*q)


# make sure the values go around in a clockwise fashion 
areaUnitCell = rhombus_area(p1, p2, p3, p4)
print(f"The area of the plaquette (rhombus) is: {areaUnitCell:.0f} nm^2")

# number_flux_perUnitCell = applied_B*1E-3 * (areaUnitCell*1E-9*1E-9) / phi_0
print(f'Filling Factor: {f:.2f}')

factor = 5 
applied_B = f * phi_0 * 1E3 / (factor* areaUnitCell*1E-9*1e-9)  
print(f'Applied B field is: {applied_B:.2f} mT, which is reduced by a factor of {factor} from expected value')


from tdgl.sources import LinearRamp, ConstantField
# Ramp the applied field 
applied_vector_potential = (
    LinearRamp(tmin=10, tmax=110)
    * ConstantField(applied_B, field_units="mT", length_units=IslandDevice.length_units)
)

#  --------------------------------  Simulation --------------------------------  #
options = tdgl.SolverOptions(
    solve_time= 2000,
    current_units="uA",
    field_units="mT",
    output_file=os.path.join(file_path, f"excitation_solution_sym_{f:.2f}.h5"),
)
excitation_solution = tdgl.solve(
    IslandDevice,
    options,
    applied_vector_potential = applied_vector_potential, 
    disorder_epsilon= epsilon_func, 
)

#  --------------------------------  Simulation Analysis --------------------------------  #

order_parameter_path = os.path.join(file_path, f'order_parameter_sym_{f:.2f}.png')
fig , ax = excitation_solution.plot_order_parameter()
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