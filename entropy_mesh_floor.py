import math
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import numpy as np
import pygame


# ---------- CONFIG ----------
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
FPS = 60

GRID_COLS = 80
GRID_ROWS = 45
DOT_BASE_RADIUS = 2
DOT_COLOR_BASE = (40, 160, 255)
BACKGROUND_COLOR = (5, 5, 12)

# Vector field dynamics
NOISE_STRENGTH = 1.5      # Base random kick strength
DAMPING = 1.3             # Pulls vectors back toward zero
MAX_SPEED = 6.0           # Clamp on vector magnitude

# Entropy control
ENTROPY_BINS = 24         # Histogram bins for magnitude distribution
MIN_SPEED_SCALE = 0.1     # When entropy is max, motion is this fraction of normal


def init_grid(cols, rows, near_z=2.0, far_z=20.0, plate_width=12.0):
    """Precompute static grid geometry in world space."""
    grid_points = []
    for j in range(rows):
        for i in range(cols):
            u = i / (cols - 1) if cols > 1 else 0.5
            v = j / (rows - 1) if rows > 1 else 0.5

            z_depth = near_z + v * (far_z - near_z)
            x_plane = (u - 0.5) * plate_width
            grid_points.append((x_plane, z_depth, u, v, i, j))

    edges = []
    for j in range(rows):
        for i in range(cols):
            idx = j * cols + i
            if i < cols - 1:  # right neighbor
                edges.append((idx, idx + 1))
            if j < rows - 1:  # neighbor further away (down)
                edges.append((idx, idx + cols))

    return grid_points, edges


def compute_entropy(field):
    """Compute normalized Shannon entropy of vector magnitudes in the field.

    Returns value in [0, 1]. Higher means more spread-out magnitudes.
    """
    mags = np.linalg.norm(field, axis=-1).ravel()
    if mags.size == 0:
        return 0.0

    max_mag = float(np.max(mags))
    if max_mag <= 1e-6:
        return 0.0

    hist, _ = np.histogram(mags, bins=ENTROPY_BINS, range=(0.0, max_mag))
    total = hist.sum()
    if total == 0:
        return 0.0

    p = hist.astype(np.float64) / total
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0

    H = -np.sum(p * np.log(p))
    H_max = math.log(ENTROPY_BINS)
    return float(H / H_max) if H_max > 0 else 0.0


def update_field(field, dt):
    """Update vector field with random kicks and damping.

    The overall update speed is modulated by entropy: when the field
    is very diverse (high entropy), motion slows down.
    """
    entropy = compute_entropy(field)
    # Map entropy in [0,1] to speed scale in [MIN_SPEED_SCALE, 1]
    speed_scale = (1.0 - entropy) * (1.0 - MIN_SPEED_SCALE) + MIN_SPEED_SCALE

    # Wave propagation: each vector is influenced by neighbors
    rows, cols, _ = field.shape
    field_new = field.copy()
    for j in range(rows):
        for i in range(cols):
            neighbors = []
            for dj, di in [(-1,0),(1,0),(0,-1),(0,1)]:
                nj, ni = j+dj, i+di
                if 0 <= nj < rows and 0 <= ni < cols:
                    neighbors.append(field[nj, ni])
            if neighbors:
                avg_neighbor = np.mean(neighbors, axis=0)
                # Blend with own vector
                field_new[j, i] = 0.7 * field[j, i] + 0.3 * avg_neighbor

    # Ornstein-Uhlenbeck-like dynamics: damp toward 0, add noise
    noise = np.random.normal(loc=0.0, scale=NOISE_STRENGTH, size=field.shape)
    field_new += dt * speed_scale * (noise - DAMPING * field_new)

    # Clamp magnitudes to avoid runaway
    mags = np.linalg.norm(field_new, axis=-1, keepdims=True)
    too_fast = mags > MAX_SPEED
    if np.any(too_fast):
        field_new[too_fast] *= MAX_SPEED / (mags[too_fast] + 1e-6)

    field[:] = field_new
    return entropy


def main():
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Entropy‑Controlled Mesh Floor")

    clock = pygame.time.Clock()

    cols = GRID_COLS
    rows = GRID_ROWS

    camera_y = 4.0
    near_z = 2.0
    far_z = 20.0
    plate_width = 12.0

    fov_deg = 60.0
    fov_rad = math.radians(fov_deg)
    proj_scale = (WINDOW_HEIGHT / 2) / math.tan(fov_rad / 2)

    grid_points, edges = init_grid(cols, rows, near_z=near_z, far_z=far_z, plate_width=plate_width)

    # Vector field: rows x cols x 2 (vx, vy) in an abstract space
    # We'll map magnitude and orientation to lift and brightness.
    field = np.zeros((rows, cols, 2), dtype=np.float32)

    font = pygame.font.SysFont(None, 24)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update field and get current entropy
        entropy = update_field(field, dt)

        screen.fill(BACKGROUND_COLOR)

        # Normalize entropy for overlay text: higher = slower
        speed_scale = (1.0 - entropy) * (1.0 - MIN_SPEED_SCALE) + MIN_SPEED_SCALE

        projected = [None] * len(grid_points)

        # Compute local entropy for each point (windowed neighborhood)
        local_entropy = np.zeros((rows, cols), dtype=np.float32)
        window = 2  # neighborhood radius
        for j in range(rows):
            for i in range(cols):
                j0 = max(0, j - window)
                j1 = min(rows, j + window + 1)
                i0 = max(0, i - window)
                i1 = min(cols, i + window + 1)
                neighborhood = field[j0:j1, i0:i1].reshape(-1, 2)
                local_entropy[j, i] = compute_entropy(neighborhood)

        for idx, (x_plane, z_depth, u, v, i, j) in enumerate(grid_points):
            wx = x_plane
            wy = 0.0
            wz = -z_depth

            vx = wx
            vy = wy - camera_y
            vz = wz

            depth = -vz
            if depth <= 0.0:
                continue

            sx = int(WINDOW_WIDTH / 2 + vx * proj_scale / depth)
            sy = int(WINDOW_HEIGHT / 2 - vy * proj_scale / depth)

            if sx < 0 or sx >= WINDOW_WIDTH or sy < 0 or sy >= WINDOW_HEIGHT:
                continue

            vec = field[j, i]
            mag = float(np.linalg.norm(vec))

            # Tremble: offset dot position by vector, scaled by entropy
            tremble_scale = (1.0 - entropy) * 1.2 + 0.2  # more tremble, less damping
            tremble_x = vec[0] * tremble_scale
            sx = int(sx + tremble_x)
            # sy remains unchanged for horizontal-only trembling

            if sx < 0 or sx >= WINDOW_WIDTH or sy < 0 or sy >= WINDOW_HEIGHT:
                continue

            depth_norm = (z_depth - near_z) / (far_z - near_z + 1e-6)
            depth_norm = max(0.0, min(1.0, depth_norm))

            # Color code: map local entropy to color (blue=low, green=mid, red=high)
            le = local_entropy[j, i]
            r = int(255 * le)
            g = int(255 * (1.0 - abs(le - 0.3) * 2))  # green at lower entropy
            b = int(255 * (1.0 - le))

            # Brightness from magnitude, softly faded by depth
            brightness = 0.15 + (mag / MAX_SPEED) * 0.7
            brightness *= (0.4 + 0.6 * (1.0 - depth_norm))
            brightness = max(0.0, min(1.0, brightness))
            r = int(r * brightness)
            g = int(g * brightness)
            b = int(b * brightness)
            # Clamp color values to valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            projected[idx] = (sx, sy, brightness, r, g, b)

        # Draw mesh lines
        for a, b in edges:
            pa = projected[a]
            pb = projected[b]
            if pa is None or pb is None:
                continue
            x1, y1, br1, r1, g1, b1 = pa
            x2, y2, br2, r2, g2, b2 = pb

            br = max(0.0, min(1.0, (br1 + br2) * 0.5)) * 0.7
            lr = int(DOT_COLOR_BASE[0] * br)
            lg = int(DOT_COLOR_BASE[1] * br)
            lb = int(DOT_COLOR_BASE[2] * br)

            pygame.draw.line(screen, (lr, lg, lb), (x1, y1), (x2, y2), 1)

        # Draw dots on top
        for p in projected:
            if p is None:
                continue
            sx, sy, brightness, r, g, b = p
            pygame.draw.circle(screen, (r, g, b), (sx, sy), DOT_BASE_RADIUS)

        # HUD text showing entropy & speed scaling
        text = f"entropy: {entropy:.3f}  speed: {speed_scale:.3f}"
        surf = font.render(text, True, (200, 200, 220))
        screen.blit(surf, (10, 10))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
