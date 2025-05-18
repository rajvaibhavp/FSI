import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
Lx = 10.0  # Domain length in x-direction (m)
Ly = 5.0   # Domain length in y-direction (m)
nx = 101   # Number of grid points in x-direction
ny = 51    # Number of grid points in y-direction
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Fluid properties
rho_f = 1.0       # Fluid density (kg/m^3)
mu = 0.01         # Dynamic viscosity (kg/(m·s))
nu = mu / rho_f   # Kinematic viscosity (m^2/s)

# Plate properties
plate_x = 2.0     # x-position of plate (m)
plate_height = 3.0 # Height of plate (m)
E = 1e4           # Young's modulus (Pa)
thickness = 0.1   # Plate thickness (m)
rho_s = 100.0     # Plate density (kg/m^3)
damping = 0.1     # Damping coefficient

# Flow conditions
U_inlet = 1.0     # Inlet velocity (m/s)

# Time parameters
dt = 0.01         # Time step (s)
total_time = 10.0  # Total simulation time (s)
nt = int(total_time / dt)  # Number of time steps

# Initialize fluid velocity fields
u = np.zeros((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity
p = np.zeros((ny, nx))  # pressure

# Initialize plate displacement
plate_nodes = int(plate_height / dy) + 1
plate_disp = np.zeros(plate_nodes)  # Displacement in x-direction
plate_vel = np.zeros(plate_nodes)   # Velocity in x-direction
plate_acc = np.zeros(plate_nodes)   # Acceleration in x-direction

# Plate position in grid indices
plate_idx = int(plate_x / dx)

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Function to solve pressure Poisson equation
def solve_pressure(u, v, p, dx, dy, dt, rho_f):
    b = np.zeros_like(p)
    b[1:-1, 1:-1] = (rho_f / (2 * dt) * 
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / dx + 
                     (v[2:, 1:-1] - v[:-2, 1:-1]) / dy))
    
    # Sparse matrix construction for Poisson equation
    A = build_poisson_matrix(nx, ny, dx, dy)
    p_flat = spsolve(A, b.flatten())
    return p_flat.reshape((ny, nx))

def build_poisson_matrix(nx, ny, dx, dy):
    N = nx * ny
    main_diag = -2/dx**2 - 2/dy**2
    x_diag = np.ones(N) / dx**2
    y_diag = np.ones(N) / dy**2
    
    # Boundary conditions (Neumann)
    # Left and right boundaries
    for j in range(ny):
        x_diag[j * nx] = 0
        x_diag[j * nx + nx - 1] = 0
    
    # Top and bottom boundaries
    for i in range(nx):
        y_diag[i] = 0
        y_diag[(ny - 1) * nx + i] = 0
    
    diagonals = [main_diag * np.ones(N), 
                 x_diag, x_diag,
                 y_diag, y_diag]
    offsets = [0, -1, 1, -nx, nx]
    
    A = diags(diagonals, offsets, shape=(N, N), format='csr')
    return A

# Function to update plate dynamics
def update_plate(plate_disp, plate_vel, plate_acc, force, dt, E, thickness, rho_s, damping):
    # Simple beam model (1D)
    n = len(plate_disp)
    k = E * thickness**3 / (12 * (dy**4))  # Simplified stiffness
    m = rho_s * thickness * dy             # Mass per node
    
    # Calculate new acceleration
    plate_acc_new = (force / m) - (k * plate_disp / m) - (damping * plate_vel / m)
    
    # Update velocity and displacement
    plate_vel_new = plate_vel + 0.5 * (plate_acc + plate_acc_new) * dt
    plate_disp_new = plate_disp + plate_vel * dt + 0.5 * plate_acc * dt**2
    
    return plate_disp_new, plate_vel_new, plate_acc_new

# Prepare for animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.3)

# Initialize plots
# Fluid plot
fluid_plot = ax1.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], scale=20)
plate_line, = ax1.plot([plate_x] * plate_nodes, y[:plate_nodes], 'k-', linewidth=2)
deformed_plate, = ax1.plot([plate_x] * plate_nodes, y[:plate_nodes], 'r--', linewidth=2)
ax1.set_title('Fluid Flow and Plate Deformation')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, Ly)

# Plate displacement plot
disp_plot, = ax2.plot(plate_disp, y[:plate_nodes], 'b-')
ax2.set_title('Plate Displacement')
ax2.set_xlabel('Displacement (m)')
ax2.set_ylabel('y (m)')
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(0, plate_height)
ax2.grid(True)

# Time display
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

# Main simulation loop
def update(frame):
    global u, v, p, plate_disp, plate_vel, plate_acc
    
    # Apply inlet boundary condition
    u[:, 0] = U_inlet
    v[:, 0] = 0
    
    # Apply no-slip boundary condition at plate
    for j in range(plate_nodes):
        u[j, plate_idx] = plate_vel[j]  # Match fluid velocity to plate velocity
        v[j, plate_idx] = 0
    
    # Solve Navier-Stokes equations (simplified)
    # Predictor step (without pressure)
    u_star = u.copy()
    v_star = v.copy()
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if j == plate_idx and i < plate_nodes:  # Skip plate nodes
                continue
                
            # X-momentum
            conv_u = u[i,j] * (u[i,j+1] - u[i,j-1]) / (2*dx) + v[i,j] * (u[i+1,j] - u[i-1,j]) / (2*dy)
            diff_u = nu * ((u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dx**2 + (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dy**2)
            u_star[i,j] = u[i,j] + dt * (-conv_u + diff_u)
            
            # Y-momentum
            conv_v = u[i,j] * (v[i,j+1] - v[i,j-1]) / (2*dx) + v[i,j] * (v[i+1,j] - v[i-1,j]) / (2*dy)
            diff_v = nu * ((v[i,j+1] - 2*v[i,j] + v[i,j-1]) / dx**2 + (v[i+1,j] - 2*v[i,j] + v[i-1,j]) / dy**2)
            v_star[i,j] = v[i,j] + dt * (-conv_v + diff_v)
    
    # Solve pressure Poisson equation
    p = solve_pressure(u_star, v_star, p, dx, dy, dt, rho_f)
    
    # Corrector step
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if j == plate_idx and i < plate_nodes:  # Skip plate nodes
                continue
                
            u[i,j] = u_star[i,j] - dt * (p[i,j+1] - p[i,j-1]) / (2 * dx * rho_f)
            v[i,j] = v_star[i,j] - dt * (p[i+1,j] - p[i-1,j]) / (2 * dy * rho_f)
    
    # Calculate fluid force on plate
    force = np.zeros(plate_nodes)
    for j in range(plate_nodes):
        # Pressure and viscous forces
        pressure_force = p[j, plate_idx] * dy
        viscous_force = mu * (u[j, plate_idx+1] - u[j, plate_idx-1]) / (2*dx) * dy
        force[j] = pressure_force + viscous_force
    
    # Update plate dynamics
    plate_disp_new, plate_vel_new, plate_acc_new = update_plate(
        plate_disp, plate_vel, plate_acc, force, dt, E, thickness, rho_s, damping)
    
    plate_disp, plate_vel, plate_acc = plate_disp_new, plate_vel_new, plate_acc_new
    
    # Update visualization
    fluid_plot.set_UVC(u[::2, ::2], v[::2, ::2])
    deformed_plate.set_data(plate_x + plate_disp, y[:plate_nodes])
    disp_plot.set_data(plate_disp, y[:plate_nodes])
    time_text.set_text(f'Time = {frame*dt:.2f} s')
    
    return fluid_plot, plate_line, deformed_plate, disp_plot, time_text

# Create animation
ani = FuncAnimation(fig, update, frames=nt, interval=50, blit=False)

plt.tight_layout()
plt.show()