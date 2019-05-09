from pyamg.gallery import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
import numpy as np
import plot

stencil = diffusion_stencil_2d(epsilon=0.001, theta = np.pi/4)
A = stencil_grid(stencil, (100, 100), dtype=float)

plot.spy(A, color='black', markersize = 0.2)
plot.save_plot('aniso_sparsity.pdf')
