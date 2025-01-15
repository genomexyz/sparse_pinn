import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Define the hill function (vectorized)
def cosine_hill(x):
    periodic_x = x % (hill_width + flat_width)  # Position within one period of hill + flat span
    hill_region = periodic_x < hill_width  # True if within hill region, False if within flat span
    reverse_hill = (x // (hill_width + flat_width)) % 2 == 1  # Alternate waves are reversed
    result = np.zeros_like(x)  # Initialize result array
    result[~reverse_hill] = hill_height / 2 * (1 + np.cos(np.pi * periodic_x[~reverse_hill] / hill_width)) * hill_region[~reverse_hill]
    result[reverse_hill] = hill_height / 2 * (1 - np.cos(np.pi * periodic_x[reverse_hill] / hill_width)) * hill_region[reverse_hill]
    return result

epsilon = 1e-10  # You can adjust this value if needed
def finite_difference_1d(f, coords, axis):
    df = np.zeros_like(f)
    delta = np.zeros_like(f)
    if axis == 0:  # Partial derivative w.r.t x
        delta[1:-1] = coords[2:] - coords[:-2]
        delta[delta == 0] = epsilon  # Replace zeros with a small value
        df[1:-1] = (f[2:] - f[:-2]) / delta[1:-1]
    elif axis == 1:  # Partial derivative w.r.t y
        delta[1:-1] = coords[2:] - coords[:-2]
        delta[delta == 0] = epsilon  # Replace zeros with a small value
        df[1:-1] = (f[2:] - f[:-2]) / delta[1:-1]
    return df

def second_derivative_1d(f, coords):
    d2f = np.zeros_like(f)
    delta = coords[1:-1] - coords[:-2]
    delta[delta == 0] = epsilon  # Replace zeros with a small value
    d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / (delta**2)
    return d2f

def finite_difference_2d(f, coords2d, axis):
        """
        Compute first-order finite difference derivatives along a given axis.
        Args:
            f: Tensor of shape (height, width).
            axis: Axis to compute the derivative (0 for y, 1 for x).
        Returns:
            df: Tensor of the same shape as f containing the derivative.
        """
        df = torch.zeros_like(f)
        #print('cek shape coords', coords2d.size())
        if axis == 0:  # Derivative along y (vertical)
            #delta = torch.clamp(f[2:, :] - f[:-2, :], min=self.epsilon)
            #delta = torch.clamp(coords[2:, :] - coords[:-2, :], min=self.epsilon)
            delta = coords2d[2:, :] - coords2d[:-2, :]
            df[1:-1, :] = (f[2:, :] - f[:-2, :]) / delta
        elif axis == 1:  # Derivative along x (horizontal)
            #delta = torch.clamp(f[:, 2:] - f[:, :-2], min=self.epsilon)
            delta = coords2d[:, 2:] - coords2d[:, :-2]
            df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / delta
        return df
    
def second_derivative_2d(f, coords2d, axis):
    """
    Compute second-order finite difference derivatives along a given axis.
    Args:
        f: Tensor of shape (height, width).
        axis: Axis to compute the derivative (0 for y, 1 for x).
    Returns:
        d2f: Tensor of the same shape as f containing the second derivative.
    """
    d2f = torch.zeros_like(f)
    if axis == 0:  # Second derivative along y (vertical)
        #delta = torch.clamp(f[2:, :] - f[:-2, :], min=self.epsilon)
        delta = coords2d[2:, :] - coords2d[:-2, :]
        d2f[1:-1, :] = (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / (delta**2)
    elif axis == 1:  # Second derivative along x (horizontal)
        #delta = torch.clamp(f[:, 2:] - f[:, :-2], min=self.epsilon)
        delta = coords2d[:, 2:] - coords2d[:, :-2]
        d2f[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (delta**2)
    return d2f

def physics_loss(predictions, coordsx, coordsy):
        """
        Args:
            predictions: Predicted outputs [u_mean, v_mean, p_mean, uv] (shape: [N, 4])
            coords: Coordinates [x, y] (shape: [N, 2])
        
        Returns:
            loss: Combined residual loss
        """
        # Extract predicted quantities
        u_mean, v_mean, p_mean, uv = predictions[:, :, 0], predictions[:, :, 1], predictions[:, :, 2], predictions[:, :, 3]

        # Extract spatial coordinates
        #x, y = coords[:, 0], coords[:, 1]
        #x = coordsx
        #y = coordsy

        #deltax = x[1] - x[0]
        #deltay = y[1] - y[0]

        # Create meshgrid
        #X, Y = torch.meshgrid(x, y, indexing='ij')  # 'ij' indexing matches NumPy's default behavior

        # Compute first derivatives
        du_dx = finite_difference_2d(u_mean, coordsx, axis=1)
        du_dy = finite_difference_2d(u_mean, coordsy, axis=0)
        dv_dx = finite_difference_2d(v_mean, coordsx, axis=1)
        dv_dy = finite_difference_2d(v_mean, coordsy, axis=0)
        dp_dx = finite_difference_2d(p_mean, coordsx, axis=1)
        dp_dy = finite_difference_2d(p_mean, coordsy, axis=0)
        d_uv_dx = finite_difference_2d(uv, coordsx, axis=1)
        d_uv_dy = finite_difference_2d(uv, coordsy, axis=0)

        # Compute second derivatives
        d2u_dx2 = second_derivative_2d(u_mean, coordsx, axis=1)
        d2u_dy2 = second_derivative_2d(u_mean, coordsy, axis=0)
        d2v_dx2 = second_derivative_2d(v_mean, coordsx, axis=1)
        d2v_dy2 = second_derivative_2d(v_mean, coordsy, axis=0)

        # Compute residuals
        epsilon_1 = du_dx + dv_dy  # Continuity equation residual

        epsilon_2 = (lambda_m * (u_mean * du_dx + v_mean * du_dy) + 
                     (1 / rho) * dp_dx - 
                     nu * (d2u_dx2 + d2u_dy2) + 
                     d_uv_dy)  # X-momentum residual

        epsilon_3 = (lambda_m * (u_mean * dv_dx + v_mean * dv_dy) + 
                     (1 / rho) * dp_dy - 
                     nu * (d2v_dx2 + d2v_dy2) + 
                     d_uv_dx)  # Y-momentum residual
        
        print('mean epsilon 1 2 3', torch.nanmean(epsilon_1), torch.nanmean(epsilon_2), torch.nanmean(epsilon_3))
        print('std epsilon 1 2 3', np.nanstd(epsilon_1.numpy()), np.nanstd(epsilon_2.numpy()), np.nanstd(epsilon_3.numpy()))

        # Compute loss (sum of squared residuals)
        residual_loss = (torch.nanmean(epsilon_1**2) + 
                         torch.nanmean(epsilon_2**2) + 
                         torch.nanmean(epsilon_3**2))

        return residual_loss

alpha = 1
Lxh = 6
Lyh = 2.024
hill_height = 1

Lx = Lxh / hill_height
Ly = Lyh / hill_height

hill_width = alpha * 2
flat_width = Lx - 2*hill_width

#data = np.loadtxt("DNS_29_Periodic_Hills/alph05-4071-2024.dat", skiprows=20)
filename = 'alph10-6-2024.dat'
data = np.loadtxt(filename, skiprows=20)
df = pd.DataFrame(data, columns = ["x","y","u_mean","v_mean","w_mean","p_mean","dissipation_mean","vorticity_mean",\
                                        "uu","vv","ww","uv","uw","vw","pp"])

# Constants
rho = 1.0  # Assume constant density
nu = 0.01  # Kinematic viscosity
lambda_m = 1.0  # Scaling factor for momentum terms

# Numerical grid
x = df["x"].values
y = df["y"].values

x_unique = np.unique(x)
y_unique = np.unique(y)

x2d = np.reshape(x, (len(y_unique), len(x_unique)))
y2d = np.reshape(y, (len(y_unique), len(x_unique)))

x2d_torch = torch.from_numpy(x2d).float()
y2d_torch = torch.from_numpy(y2d).float()

# Flow quantities
u = df["u_mean"].values
v = df["v_mean"].values
p = df["p_mean"].values
uv = df["uv"].values

u[y < hill_height] = np.nan
v[y < hill_height] = np.nan
p[y < hill_height] = np.nan
uv[y < hill_height] = np.nan

#u_torch = torch.from_numpy(u)
#v_torch = torch.from_numpy(v)
#p_torch = torch.from_numpy(p)
#uv_torch = torch.from_numpy(uv)

test_label = [u, v, p, uv]
test_label = np.array(test_label)
test_label = test_label.T
test_label_torch = torch.from_numpy(test_label).float()

#test ploss
label2d = test_label_torch.view(len(y_unique), len(x_unique), -1)
loss_testing = physics_loss(label2d, x2d_torch, y2d_torch)
print(loss_testing)