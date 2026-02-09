import sys
import math
import sys
import math
import os
import tempfile
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import numpy as np
import pygame
from pydub import AudioSegment
# ---------- CONFIG ----------
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540
FPS = 60

DEFAULT_AUDIO_PATH = "audio.wav"

GRID_COLS = 80
GRID_ROWS = 45
DOT_BASE_RADIUS = 2
DOT_COLOR_BASE = (40, 160, 255)
BACKGROUND_COLOR = (2, 2, 8)

FFT_WINDOW_SIZE = 2048


def load_audio(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        segment = AudioSegment.from_file(path)
    except Exception as e:
        raise RuntimeError(
            f"Could not decode audio file '{path}'. "
            f"For formats like .m4a or .webm, ensure FFmpeg is installed.\n{e}"
        )

    if segment.channels > 1:
        segment = segment.set_channels(1)

    sample_rate = segment.frame_rate
    sample_width = segment.sample_width

    samples = np.array(segment.get_array_of_samples())

    if sample_width == 1:
        samples = samples.astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif sample_width == 3:
        samples = samples.astype(np.float32) / (2 ** 23)
    elif sample_width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:
        samples = samples.astype(np.float32)
        max_abs = float(np.max(np.abs(samples)) or 1.0)
        samples /= max_abs

    max_val = float(np.max(np.abs(samples)) or 1.0)
    if max_val > 1.0:
        samples /= max_val

    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".ogg"):
        playback_path = path
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()
        segment.export(tmp_path, format="wav")
        playback_path = tmp_path

    return sample_rate, samples, playback_path


def get_audio_slice(audio_data, sr, t_ms, window_size):
    t_sec = t_ms / 1000.0
    center_index = int(t_sec * sr)
    start = max(0, center_index - window_size // 2)
    end = min(len(audio_data), start + window_size)
    slice_data = audio_data[start:end]

    if len(slice_data) < window_size:
        pad_width = window_size - len(slice_data)
        slice_data = np.pad(slice_data, (0, pad_width))

    return slice_data


def compute_spectrum(audio_slice):
    """Return overall amplitude, low/mid/high bands, and normalized spectrum (0..1)."""
    amp = float(np.mean(np.abs(audio_slice)))

    window = np.hanning(len(audio_slice))
    spectrum = np.fft.rfft(audio_slice * window)
    mag = np.abs(spectrum)

    # Compress huge dynamic range and reduce bass dominance
    mag = np.log1p(mag)

    # Gently boost higher bins so mids/highs are more visible
    n_bins = len(mag)
    if n_bins > 1:
        bin_pos = np.linspace(0.0, 1.0, n_bins)  # 0 = low freq, 1 = high
        # Weight from ~0.3 at bass up to ~1.0 at treble
        weights = 0.3 + 0.7 * (bin_pos ** 0.7)
        mag_weighted = mag * weights
    else:
        mag_weighted = mag

    if len(mag_weighted) < 3:
        mag_norm = mag_weighted / (np.max(mag_weighted) + 1e-6) if len(mag_weighted) > 0 else mag_weighted
        return amp, (amp, amp, amp), mag_norm

    n = len(mag_weighted)
    third = n // 3
    low = float(np.mean(mag_weighted[:third]))
    mid = float(np.mean(mag_weighted[third:2 * third]))
    high = float(np.mean(mag_weighted[2 * third:]))

    total = low + mid + high + 1e-6
    low /= total
    mid /= total
    high /= total

    mag_norm = mag_weighted / (np.max(mag_weighted) + 1e-6)

    return amp, (low, mid, high), mag_norm


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.isabs(arg_path):
            audio_path = arg_path
        else:
            audio_path = os.path.join(script_dir, arg_path)
    else:
        audio_path = os.path.join(script_dir, DEFAULT_AUDIO_PATH)

    try:
        sr, audio_data, playback_path = load_audio(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    pygame.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Audio Reactive Mesh Floor")

    clock = pygame.time.Clock()

    try:
        pygame.mixer.music.load(playback_path)
    except pygame.error as e:
        print(f"Pygame could not play this file. Make sure it's a WAV file. Error: {e}")
        return

    pygame.mixer.music.play()

    cols = GRID_COLS
    rows = GRID_ROWS

    # Camera higher and farther back, less tilt for non-overlapping layers
    near_z = 2.0
    far_z = 32.0
    plate_width = 16.0
    camera_y = 12.0  # lower so mesh is centered
    camera_x = 0.0   # centered
    fov_deg = 68.0   # wider field of view
    fov_rad = math.radians(fov_deg)
    proj_scale = (WINDOW_HEIGHT / 2) / math.tan(fov_rad / 2)

    # Precompute grid points and neighbor edges for mesh lines
    grid_points = []
    for j in range(rows):
        for i in range(cols):
            u = i / (cols - 1) if cols > 1 else 0.5
            v = j / (rows - 1) if rows > 1 else 0.5

            # Rotate mesh 90 degrees to the right
            z_depth = far_z - v * (far_z - near_z)
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

    running = True
    smoothed_spectrum = None
    smoothed_amp = 0.0

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        t_ms = pygame.mixer.music.get_pos()
        if t_ms == -1:
            running = False
            continue

        audio_slice = get_audio_slice(audio_data, sr, t_ms, FFT_WINDOW_SIZE)
        amp, _, spectrum = compute_spectrum(audio_slice)

        # Smooth amplitude over time for gentler global motion
        amp_smooth_factor = 0.15
        smoothed_amp = (1.0 - amp_smooth_factor) * smoothed_amp + amp_smooth_factor * amp
        amp_boost = min(1.0, smoothed_amp * 20.0)

        screen.fill(BACKGROUND_COLOR)

        # Smooth spectrum over time to avoid jittery band movement
        if smoothed_spectrum is None or len(smoothed_spectrum) != len(spectrum):
            smoothed_spectrum = spectrum.copy()
        else:
            spec_smooth_factor = 0.2
            smoothed_spectrum = (
                (1.0 - spec_smooth_factor) * smoothed_spectrum
                + spec_smooth_factor * spectrum
            )

        spec_len = len(smoothed_spectrum)
        if spec_len == 0:
            pygame.display.flip()
            continue

        # Normalize spectrum once per frame and enhance contrast so
        # peaks stand out more; this reduces the "flat" look.
        spec = smoothed_spectrum
        # Slightly compress very low values, then apply a power curve
        # so strong frequencies dominate more while preserving relative
        # band structure for coherent frequency layout.
        spec = np.clip(spec - 0.02, 0.0, None)
        spec = np.power(spec, 1.3)
        max_spec = float(np.max(spec)) if spec_len > 0 else 0.0
        if max_spec <= 1e-6:
            max_spec = 1.0

        # Store projected positions and brightness for each point
        projected = [None] * len(grid_points)

        for idx, (x_plane, z_depth, u, v, i, j) in enumerate(grid_points):
            wx = x_plane
            wy = 0.0
            wz = -z_depth

            # Diagonal camera: offset x and y
            vx = wx
            vy = wy - camera_y
            vz = wz

            depth = -vz
            if depth <= 0.0:
                continue

            sx = int(WINDOW_WIDTH / 2 + vx * proj_scale / depth)
            # Center mesh vertically in window
            sy = int(WINDOW_HEIGHT * 0.55 - vy * proj_scale / depth)

            if sx < 0 or sx >= WINDOW_WIDTH or sy < 0 or sy >= WINDOW_HEIGHT:
                continue

            # Relate dots to frequencies with a perceptually coherent
            # log-frequency mapping from ~50 Hz to Nyquist across the mesh.
            if spec_len > 1:
                nyquist = sr / 2.0
                f_min = 20.0
                f_max = max(f_min * 2.0, nyquist)
                log_min = math.log10(f_min)
                log_max = math.log10(f_max)
                # Map horizontal coordinate u in [0,1] to log frequency
                f_target_log = log_min + (log_max - log_min) * u
                f_target = 10.0 ** f_target_log
                f_target = max(0.0, min(nyquist, f_target))
                bin_pos = f_target / nyquist * (spec_len - 1)
                center_bin = int(round(bin_pos))
                # Average a small neighborhood so each column reflects a
                # band of nearby frequencies rather than a single bin.
                window_radius = 2
                num = 0.0
                den = 0.0
                for dj in range(-window_radius, window_radius + 1):
                    b = center_bin + dj
                    if b < 0 or b >= spec_len:
                        continue
                    val = float(spec[b])
                    if val <= 0.0:
                        continue
                    # Gaussian weight in bin space for smooth blending
                    w = math.exp(-0.5 * (dj * dj) / (1.5 * 1.5))
                    num += val * w
                    den += w
                if den > 1e-6:
                    raw_level = num / den
                else:
                    raw_level = 0.0
            else:
                raw_level = float(spec[0]) if spec_len > 0 else 0.0
            level = max(0.0, min(1.0, raw_level / max_spec))

            # Make mesh curves more visible: boost vertical lift and contrast
            vertical_span = WINDOW_HEIGHT * 0.7 * amp_boost
            # Sharper frequency contrast
            level = level ** 1.25
            lift = level * vertical_span
            sy = int(sy - lift)

            if sy < 0 or sy >= WINDOW_HEIGHT:
                continue

            depth_norm = (z_depth - near_z) / (far_z - near_z + 1e-6)
            depth_norm = max(0.0, min(1.0, depth_norm))

            # Always-lit dots: brightness never goes to zero, just scales with level
            base_brightness = 0.2  # minimum so nothing turns off
            var_brightness = level * (0.6 + 0.4 * amp_boost)
            brightness = (base_brightness + var_brightness)
            # Depth fade so far mesh doesn't clutter (keep current strength)
            brightness *= (0.4 + 0.6 * (1.0 - depth_norm))
            brightness = max(0.0, min(1.0, brightness))

            r = int(DOT_COLOR_BASE[0] * brightness)
            g = int(DOT_COLOR_BASE[1] * brightness)
            b = int(DOT_COLOR_BASE[2] * brightness)

            projected[idx] = (sx, sy, brightness, r, g, b)

        # Draw mesh lines between neighboring visible points
        for a, b in edges:
            pa = projected[a]
            pb = projected[b]
            if pa is None or pb is None:
                continue
            x1, y1, br1, r1, g1, b1 = pa
            x2, y2, br2, r2, g2, b2 = pb

            # Dimmer lines in base mesh color
            br = max(0.0, min(1.0, (br1 + br2) * 0.5)) * 0.6
            lr = int(DOT_COLOR_BASE[0] * br)
            lg = int(DOT_COLOR_BASE[1] * br)
            lb = int(DOT_COLOR_BASE[2] * br)

            pygame.draw.line(screen, (lr, lg, lb), (x1, y1), (x2, y2), 1)

        # Draw dots on top of lines
        for p in projected:
            if p is None:
                continue
            sx, sy, brightness, r, g, b = p
            radius = DOT_BASE_RADIUS
            pygame.draw.circle(screen, (r, g, b), (sx, sy), max(1, int(radius)))

        pygame.display.flip()

    pygame.mixer.music.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
