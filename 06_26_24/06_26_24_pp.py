

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
total_height = 4000
width = 3500
height = width
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