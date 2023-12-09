import numpy as np


def xy_pixel_indexes(img, swap=False):
    height, width = img.shape[:2]
    xs = np.array([range(width)] * height).astype(np.float32)
    ys = np.array([[y for _ in range(width)] for y in range(height)]).astype(np.float32)

    if swap:
        return ys, xs, height, width
    else:
        return xs, ys, width, height


def quadratic_formulae(a, b, c):
    # should I use np.root instead? mind that a is an array
    vs = np.sqrt(b ** 2 - 4 * a * c)
    a2 = a * 2
    return (-b - vs) / a2, (-b + vs) / a2


def debug_draw_outer_cylinder_proj(z0, r, f, o):
    """
    Debug remap_outer_cylinder
    """
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    draw_circle = plt.Circle((0, z0), r, fill=False)
    axes.set_aspect(1)
    plt.xlim(-r * 1.5, r * 1.5)
    plt.ylim(0, f + r * 2)
    axes.add_artist(draw_circle)
    x = np.linspace(-r * 1.15, r * 1.15, 2)
    z = f / o * x  # function
    plt.plot(x, z)
    plt.plot(x, -z)
    x = np.linspace(-o, o, 2)
    plt.plot(x, [f] * len(x))
    plt.title('Debug Cylinder Projection')
    plt.savefig('plot_cylinder_projection.png')


def remap_inner_cylinder(src, _, fov=90, swap_xy=False):
    """
    Adapted from https://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection ;
    radius always equal to w; and f changes according to fov.
    @param xs: final positions x component; most cases, just index original image x pixels
    @param ys: final positions y component; most cases, just index original image y pixels
    @param w: width of the image
    @param h: height of the image
    @param fov: field of view in degrees
    @return: x and y mappings ( to the original image pixels )
    """
    xs, ys, w, h = xy_pixel_indexes(src, swap_xy)

    pc = [xs - w / 2, ys - h / 2]  # offset center

    ratio = np.tan(np.deg2rad(fov / 2))
    omega = w / 2
    f = omega / ratio
    r = w

    z0 = f - np.sqrt(r * r - omega * omega)
    _, zc = quadratic_formulae((pc[0] ** 2) / (f ** 2) + 1, -2 * z0, z0 ** 2 - r ** 2)

    final_point = [pc[0] * zc / f, pc[1] * zc / f]
    final_point[0] += w / 2
    final_point[1] += h / 2

    if swap_xy:
        return final_point[1], final_point[0]
    else:
        return final_point[0], final_point[1]


def remap_outer_cylinder(src, _, fov=90, swap_xy=False):
    """
    Map to a cylinder far away.
    The cylinder radius and position is set to edge the imaginary horizontal view frustum
      while having its lowest point at the position (x = 0, z = f);
      thus keeping the x centered pixels with the same height.

    If this behavior is not clear, use the commented debug_draw for a top view of the mapping.

    @param xs: final positions x component; most cases, just index original image x pixels
    @param ys: final positions y component; most cases, just index original image y pixels
    @param w: width of the image
    @param h: height of the image
    @param fov: field of view in degrees
    @return: x and y mappings ( to the original image pixels )
    """
    xs, ys, w, h = xy_pixel_indexes(src, swap_xy)

    pc = [xs - w / 2, ys - h / 2]  # offset center

    ratio = np.tan(np.deg2rad(fov / 2))
    omega = w / 2
    f = omega / ratio

    m = f / omega; k = m + m ** -1; n = 1 - (1 / (1 + m ** -2))
    _, z0 = quadratic_formulae(a=1 - k ** -2 - n ** 2, b=-2 * f, c=f ** 2)
    r = z0 - f

    # debug_draw_outer_cylinder_proj(z0, r, f, omega)  # if the behavior is not clear, draw this

    zc, _ = quadratic_formulae((pc[0] ** 2) / (f ** 2) + 1, -2 * z0, z0 ** 2 - r ** 2)
    final_point = [pc[0] * zc / f, pc[1] * zc / f]
    final_point[0] += w / 2
    final_point[1] += h / 2
    if swap_xy:
        return final_point[1], final_point[0]
    else:
        return final_point[0], final_point[1]


