import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized
from numerical import difference, operator

class Fluid:
    def __init__(self, shape, *quantities, pressure_order=1, advect_order=3, diffusion_rate=0.1):
        self.shape = shape
        self.dimensions = len(shape)
        self.quantities = quantities
        self.advect_order = advect_order
        self.diffusion_rate = diffusion_rate

        self._initialize_fields()
        self._initialize_pressure_solver(pressure_order)

    def _initialize_fields(self):
        for q in self.quantities:
            setattr(self, q, np.zeros(self.shape))
        self.indices = np.indices(self.shape)
        self.velocity = np.zeros((self.dimensions, *self.shape))

    def _initialize_pressure_solver(self, pressure_order):
        laplacian = operator(self.shape, difference(2, pressure_order))
        self.pressure_solver = factorized(laplacian)

    def step(self):
        self.euler()
        self.diffusion()

    def euler(self):
        advection_map = self.indices - self.velocity
        self._advect_fields(advection_map)
        divergence, curl, pressure = self._compute_pressure(advection_map)
        self._apply_pressure(pressure)

    def _advect_fields(self, advection_map):
        for d in range(self.dimensions):
            self.velocity[d] = self._advect(self.velocity[d], advection_map)
        for q in self.quantities:
            setattr(self, q, self._advect(getattr(self, q), advection_map))

    def _advect(self, field, advection_map, filter_epsilon=10e-2, mode='constant'):
        filtered = spline_filter(field, order=self.advect_order, mode=mode)
        field = filtered * (1 - filter_epsilon) + field * filter_epsilon
        return map_coordinates(field, advection_map, prefilter=False, order=self.advect_order, mode=mode)

    def _compute_pressure(self, advection_map):
        jacobian_shape = (self.dimensions,) * 2
        partials = tuple(np.gradient(d) for d in self.velocity)
        jacobian = np.stack(partials).reshape(*jacobian_shape, *self.shape)
        divergence = jacobian.trace()
        curl = self._compute_curl(jacobian, jacobian_shape)
        pressure = self.pressure_solver(divergence.flatten()).reshape(self.shape)
        return divergence, curl, pressure

    def _compute_curl(self, jacobian, jacobian_shape):
        curl_mask = np.triu(np.ones(jacobian_shape, dtype=bool), k=1)
        return (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()

    def _apply_pressure(self, pressure):
        self.velocity -= np.gradient(pressure)

    def diffusion(self):
        for q in self.quantities:
            field = getattr(self, q)
            laplacian = self._compute_laplacian(field)
            field += self.diffusion_rate * laplacian
            setattr(self, q, field)

    def _compute_laplacian(self, field):
        return np.sum([np.roll(field, shift, axis) for shift in (-1, 1) for axis in range(self.dimensions)], axis=0) - 4 * field