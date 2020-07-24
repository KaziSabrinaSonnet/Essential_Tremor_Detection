#%%
time=np.array([9, 8, 7, 6, 5, 4, 3])
volts=np.array([1, 1, 1, 2, 2, 3, 1]) 
#%%
def masks(vec):
    d = np.diff(vec)
    dd = np.diff(d)

    # Mask of locations where graph goes to vertical or horizontal, depending on vec
    to_mask = ((d[:-1] != 0) & (d[:-1] == -dd))
    # Mask of locations where graph comes from vertical or horizontal, depending on vec
    from_mask = ((d[1:] != 0) & (d[1:] == dd))
    return to_mask, from_mask
#%%
def apply_mask(mask, x, y):
    return x[1:-1][mask], y[1:-1][mask]

to_vert_mask, from_vert_mask = masks(time)
to_horiz_mask, from_horiz_mask = masks(volts)
to_vert_t, to_vert_v = apply_mask(to_vert_mask, time, volts)
from_vert_t, from_vert_v = apply_mask(from_vert_mask, time, volts)
to_horiz_t, to_horiz_v = apply_mask(to_horiz_mask, time, volts)
from_horiz_t, from_horiz_v = apply_mask(from_horiz_mask, time, volts)

plt.plot(time, volts, 'b-')
plt.plot(to_vert_t, to_vert_v, 'r^', label='Plot goes vertical')
plt.plot(from_vert_t, from_vert_v, 'kv', label='Plot stops being vertical')
plt.plot(to_horiz_t, to_horiz_v, 'r>', label='Plot goes horizontal')
plt.plot(from_horiz_t, from_horiz_v, 'k<', label='Plot stops being horizontal')
plt.legend()
plt.show()

# %%
import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import pyaC

# Generate some 'data'
x = np.arange(100.)**2
y = np.sin(x)

# Set the last data point to zero.
# It will not be counted as a zero crossing!
y[-1] = 0

# Set point to zero. This will be counted as a
# zero crossing
y[10] = 0.0

# Get coordinates and indices of zero crossings
xc, xi = pyaC.zerocross1d(x, y, getIndices=True)

# Plot the data
plt.plot(x, y, 'b.-')
# Add black points where the zero line is crossed
plt.plot(xc, np.zeros(len(xc)), 'kp')
# Add green points at data points preceding an actual
# zero crossing.
plt.plot(x[xi], y[xi], 'gp')
plt.show()

# %%
