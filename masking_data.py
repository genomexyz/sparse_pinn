import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the hill function (vectorized)
def cosine_hill(x):
    periodic_x = x % (hill_width + flat_width)  # Position within one period of hill + flat span
    hill_region = periodic_x < hill_width  # True if within hill region, False if within flat span
    reverse_hill = (x // (hill_width + flat_width)) % 2 == 1  # Alternate waves are reversed
    result = np.zeros_like(x)  # Initialize result array
    result[~reverse_hill] = hill_height / 2 * (1 + np.cos(np.pi * periodic_x[~reverse_hill] / hill_width)) * hill_region[~reverse_hill]
    result[reverse_hill] = hill_height / 2 * (1 - np.cos(np.pi * periodic_x[reverse_hill] / hill_width)) * hill_region[reverse_hill]
    return result

filename = 'DNS_29_Periodic_Hills/alph10-6-2024.dat'

#data = np.loadtxt("DNS_29_Periodic_Hills/alph05-4071-2024.dat", skiprows=20)
data = np.loadtxt(filename, skiprows=20)
df = pd.DataFrame(data, columns = ["x","y","u_mean","v_mean","w_mean","p_mean","dissipation_mean","vorticity_mean",\
                                        "uu","vv","ww","uv","uw","vw","pp"])

alpha = 1
Lxh = 6
Lyh = 2.024
hill_height = 1

Lx = Lxh / hill_height
Ly = Lyh / hill_height

hill_width = alpha * 2
flat_width = Lx - 2*hill_width

all_x = df["x"].values
all_y = df["y"].values

hill_height = cosine_hill(all_x)

u_mean = df["u_mean"].values
v_mean = df["v_mean"].values
p_mean = df["p_mean"].values

uu = df['uu'].values
vv = df['vv'].values
ww = df['ww'].values

u_mean[all_y < hill_height] = np.nan
v_mean[all_y < hill_height] = np.nan
p_mean[all_y < hill_height] = np.nan

uu[all_y < hill_height] = np.nan
vv[all_y < hill_height] = np.nan
ww[all_y < hill_height] = np.nan

# Example 1: Plotting the mean velocity (u_mean) as a function of x and y
plt.figure(figsize=(10, 6))
sc = plt.scatter(df["x"], df["y"], c=u_mean, cmap="viridis", s=5, edgecolor='none')
plt.colorbar(sc, label="Mean Velocity (u_mean)")
plt.title("Mean Velocity (u_mean) Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Example 2: Visualising the mean pressure (p_mean)
plt.figure(figsize=(10, 6))
sc = plt.scatter(df["x"], df["y"], c=p_mean, cmap="plasma", s=5, edgecolor='none')
plt.colorbar(sc, label="Mean Pressure (p_mean)")
plt.title("Mean Pressure (p_mean) Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Example 3: Plotting turbulent kinetic energy (TKE) as a derived quantity
# TKE = 0.5 * (uu + vv + ww)
df["TKE"] = 0.5 * (df["uu"] + df["vv"] + df["ww"])

plt.figure(figsize=(10, 6))
sc = plt.scatter(df["x"], df["y"], c=df["TKE"], cmap="coolwarm", s=5, edgecolor='none')
plt.colorbar(sc, label="Turbulent Kinetic Energy (TKE)")
plt.title("Turbulent Kinetic Energy Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()