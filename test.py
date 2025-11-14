import numpy as np
import matplotlib.pyplot as plt

def create_medium(size, gradient_strength):
    """Creates a 2D refractive index grid with a vertical gradient."""
    grid = np.zeros((size, size))
    for i in range(size):
        grid[i, :] = 1.0 + gradient_strength * (i / size)
    return grid

# Example: 100x100 medium with a linear gradient
medium_size = 10
n_grid = create_medium(medium_size, 0.5)


def trace_bending_ray(start_pos, end_pos, n_grid, num_steps, learning_rate=0.2):
    """
    Simulates a ray path using a simplified discrete pseudo-bending approach.
    This version iteratively adjusts the ray path to minimize travel time.
    """
    # Initialize the ray path as a straight line
    path = np.linspace(start_pos, end_pos, num_steps)
    paths = [path]

    for iteration in range(200):  # Run several iterations for convergence
        new_path = path.copy()
        print(np.around(new_path.T, 2))
        for i in range(1, num_steps - 1):
            # Get the current and neighboring points
            p_prev, p_curr, p_next = path[i - 1], path[i], path[i + 1]

            # Simplified travel time approximation for the current segment
            segment1_len = np.linalg.norm(p_curr - p_prev)
            segment2_len = np.linalg.norm(p_next - p_curr)

            # Get average refractive index for the segments
            n1 = n_grid[int(p_prev[1]) % medium_size, int(p_prev[0]) % medium_size]
            n2 = n_grid[int(p_curr[1]) % medium_size, int(p_curr[0]) % medium_size]

            # Travel time for the current point's influence
            t_curr = segment1_len * n1 + segment2_len * n2

            # Calculate a simplified "force" or gradient direction
            # pointing towards lower refractive index, mimicking bending
            gradient = (n_grid[int(p_curr[1]) % medium_size, int(p_curr[0]) % medium_size] -
                        n_grid[int(p_curr[1] - 1) % medium_size, int(
                            p_curr[0] - 1) % medium_size])  # Example gradient calculation
            # print(gradient)

            # Simple update rule: move the point in the direction of the gradient
            # to simulate bending towards higher refractive index (lower velocity)
            bending_vector = np.array([-gradient, gradient]) * learning_rate
            new_path[i] += bending_vector
        paths.append(new_path)
        path = new_path
    return path, paths


# Define start and end points for the ray
start_point = np.array([1.0, 1.0])
end_point = np.array([9.0, 9.0])

# Trace the ray
ray_path, ray_paths = trace_bending_ray(start_point, end_point, n_grid, 100)


plt.imshow(n_grid, cmap='viridis', origin='lower', extent=[0., medium_size, 0., medium_size])
plt.colorbar(label='Refractive Index (n)')
for idx, path in enumerate(ray_paths):
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1, label='Ray Path', alpha=idx/len(ray_paths))
plt.plot(start_point[0], start_point[1], 'go', markersize=8, label='Start')
plt.plot(end_point[0], end_point[1], 'bo', markersize=8, label='End')
plt.title('Discrete Ray Bending Simulation')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
# plt.legend()
plt.show()
