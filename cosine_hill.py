import numpy as np
import matplotlib.pyplot as plt

# Parameters for the hill
H = 1  # Maximum height of the hill
L = 2  # Length of the hill
flat_span = 2  # Length of the flat span between hills

# Generate x values
x = np.linspace(0, 2*L + flat_span, 1000)  # x spans two hills with a flat region

# Define the hill function (vectorized)
def cosine_hill(x):
    periodic_x = x % (L + flat_span)  # Position within one period of hill + flat span
    hill_region = periodic_x < L  # True if within hill region, False if within flat span
    reverse_hill = (x // (L + flat_span)) % 2 == 1  # Alternate waves are reversed
    result = np.zeros_like(x)  # Initialize result array
    result[~reverse_hill] = H / 2 * (1 + np.cos(np.pi * periodic_x[~reverse_hill] / L)) * hill_region[~reverse_hill]
    result[reverse_hill] = H / 2 * (1 - np.cos(np.pi * periodic_x[reverse_hill] / L)) * hill_region[reverse_hill]
    return result

# Calculate y values
y = cosine_hill(x)

# Plot the hills with flat span
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Alternating Cosine Hills', color='blue')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # x-axis
plt.axvline(L, color='red', linewidth=0.8, linestyle='--', label='Hill Peak')  # Example hill peak

# Add labels and title
plt.title('Periodic Alternating Cosine Hills with Flat Span', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
