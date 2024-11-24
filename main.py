import numpy as np
from PIL import Image
from fluid import Fluid
from utils import create_inflow

RESOLUTION = 500, 500
DURATION = 200

INFLOW_POSITION = (250, 450)
INFLOW_RADIUS = 10
INFLOW_VELOCITY = 1

def main():
    print('Generating fluid solver, this may take some time.')
    fluid = Fluid(RESOLUTION, 'smoke')

    inflow_velocity, inflow_smoke = create_inflow(fluid, INFLOW_POSITION, INFLOW_RADIUS, INFLOW_VELOCITY)

    frames = simulate(fluid, inflow_velocity, inflow_smoke, DURATION)
    save_simulation(frames, 'smoke_simulation.gif')

def simulate(fluid, inflow_velocity, inflow_smoke, duration):
    frames = []
    for f in range(duration):
        print(f'Computing frame {f + 1} of {duration}.')
        fluid.velocity += inflow_velocity
        fluid.smoke += inflow_smoke
        fluid.step()
        frames.append(create_frame(fluid.smoke))
    return frames

def create_frame(smoke):
    color = np.dstack((smoke, smoke, smoke))
    color = (np.clip(color, 0, 1) * 255).astype('uint8')
    return Image.fromarray(color, mode='RGB')

def save_simulation(frames, filename):
    print('Saving simulation result.')
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=20, loop=0)

if __name__ == '__main__':
    main()