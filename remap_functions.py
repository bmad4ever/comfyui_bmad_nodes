import numpy as np
import cv2 as cv


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
        return final_point[1], final_point[0], None
    else:
        return final_point[0], final_point[1], None


def remap_outer_cylinder(src, _, fov=90, swap_xy=False):
    """
    Map to a cylinder far away.
    The cylinder radius and position is set to edge the imaginary horizontal view frustum
      while having its lowest point at the position (x = 0, z = f);
      thus keeping the x centered pixels with the same height.

    If this behavior is not clear, use the commented debug_draw for a top view of the mapping.

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
        return final_point[1], final_point[0], None
    else:
        return final_point[0], final_point[1], None


def get_quadratic_curve_coeffs(p0, p1, p2):
    """
    Estimate quadratic curve coefficients from the given points: p0, p1, p2
    @return: coefficienst: [ a, b, c ]
    """
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2

    a = np.array([[x1 ** 2, x1, 1], [x2 ** 2, x2, 1], [x3 ** 2, x3, 1]])
    b = np.array([y1, y2, y3])
    return np.linalg.solve(a, b)


def remap_inside_parabolas(src, _, roi_points_img, recalled=False):
    """
    @param roi_points_img: dst sized mask with the 6 points annotated
    @param recalled: safeguard against infinite recursive calls.
    @return: @return: x and y mappings ( to the original image pixels ) and roi bounding box coordinates
    """

    # region parse roi points
    contours, _ = cv.findContours(roi_points_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) != 6:
        raise(f"Can only compute with exactly 6 points but {len(contours)} were found!")

    moments = [cv.moments(c) for c in contours]
    centers = np.array([[int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])] for m in moments]).astype(np.float32)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, km_centers = cv.kmeans(centers, 2, None, criteria, 10, flags)

    zero_label_highest_y = km_centers[0, 1] > km_centers[1, 1]
    c1 = sorted(centers[labels[:, 0] == (not zero_label_highest_y)], key=lambda c: c[0])
    c2 = sorted(centers[labels[:, 0] == zero_label_highest_y], key=lambda c: c[0])

    km_centers_v = km_centers[0] - km_centers[1]
    if abs(km_centers_v[0]) > abs(km_centers_v[1]):
        if not recalled:  # shouldn't get stuck, if equal proceeds; but will use a safeguard against potential oversights!
            xs, ys, bb = remap_inside_parabolas(cv.rotate(src, cv.ROTATE_90_CLOCKWISE),
                                                _, cv.flip(cv.rotate(roi_points_img, cv.ROTATE_90_CLOCKWISE), 1),
                                                True)
            bb = (bb[1], bb[0], bb[3], bb[2])  # fix bb
            return np.rot90(np.fliplr(ys)), np.rot90(np.fliplr(xs)), bb
        raise("The current implementation only supports parabolas with no x repeated points;"
              " i.e, at no point a vertical line should cross two points of the parabola.")

    # kmeans may fail to group the points correctly when they are close to each other
    # TODO: implement a more robust alternative
    #   idea 1: draw 2 lines in the mask and fetch the points from their contours
    #     could try to fit to more points instead of 3... but what if line is jittery?
    #   idea 2: use 2 masks, one for each parabola

    zero_label_highest_y = km_centers[0, 1] > km_centers[1, 1]
    c1 = sorted(centers[labels[:, 0] == (not zero_label_highest_y)], key=lambda c: c[0])
    c2 = sorted(centers[labels[:, 0] == zero_label_highest_y], key=lambda c: c[0])

    # get upper and lower bb leeway
    lower_limit_leeway = max(c1, key=lambda c: c[1])[1] - min(c1, key=lambda c: c[1])[1]
    upper_limit_leeway = max(c2, key=lambda c: c[1])[1] - min(c2, key=lambda c: c[1])[1]
    bb = [
        int(min([c[0] for c in centers]))
        , int(min([c[1] for c in centers]) - upper_limit_leeway)
        , int(max([c[0] for c in centers]))
        , int(max([c[1] for c in centers]) + lower_limit_leeway)
    ]
    # clip lower and upper bounds (within dst)
    bb[1] = max(bb[1], 0)
    bb[3] = min(bb[3], roi_points_img.shape[0])

    origin = [bb[0], bb[1]]
    dst_box_height = int(bb[3] - bb[1])
    dst_box_width = int(bb[2] - bb[0])
    #endregion

    # get the quadratic curves coefficients from the parsed points
    c1_coeffs = get_quadratic_curve_coeffs(*c1)
    c2_coeffs = get_quadratic_curve_coeffs(*c2)

    #region auxiliary functions definitions
    def compute_w(x, y):
        """
        y = (w*a1 + w*(a2-w))x^2 + (w*b1 + w*(b2-w))x + (w*c1 + w*(c2-w)
        rearranged to compute w
        """
        a_1, b_1, c_1 = c1_coeffs
        a_2, b_2, c_2 = c2_coeffs
        return -(y - a_2 * x ** 2 - b_2 * x - c_2) / ((a_2 - a_1) * x ** 2 + (b_2 - b_1) * x + c_2 - c_1)

    def alen_integral(x, a, b):
        return (((2 * a * x + b) ** 2 + 1) ** (1 / 2) * (2 * a * x + b) + np.arcsinh(2 * a * x + b)) / (4 * a)

    def arc_len(w, x1, x2):
        a_b_s = w[:, :, np.newaxis] * c1_coeffs[:2] + (1 - w[:, :, np.newaxis]) * c2_coeffs[:2]
        a = a_b_s[:, :, 0]
        b = a_b_s[:, :, 1]
        return alen_integral(x2, a, b) - alen_integral(x1, a, b)

    def compute_x_norm(ws, px):
        p0x, p0y = c1[0]
        p2x, p2y = c1[2]
        p3x, p3y = c2[0]
        p5x, p5y = c2[2]
        leftmost_xs = ws * p0x + (1 - ws) * p3x
        rightmost_xs = ws * p2x + (1 - ws) * p5x
        full_segs_lens = arc_len(ws, leftmost_xs, rightmost_xs)
        to_point_lens = arc_len(ws, leftmost_xs, px)

        # clip sections outside roi
        to_point_lens[to_point_lens > full_segs_lens] = -10
        to_point_lens[to_point_lens < 0] = -10

        return to_point_lens / full_segs_lens
    #endregion

    # compute ys
    ys_norm = np.array([[compute_w(x + origin[0], y + origin[1]) for x in range(dst_box_width)] for y in
                        range(dst_box_height)]).astype(np.float32)
    # clip sections outside roi
    ys_norm[ys_norm > 1] = -10
    ys_norm[ys_norm < 0] = -10

    # compute xs
    xs_pxs = np.array([[x + origin[0] for x in range(dst_box_width)] for y in range(dst_box_height)]).astype(np.float32)
    xs_norm = compute_x_norm(ys_norm, xs_pxs)
    # outside roi clip is already done by compute_x_norm in previous step

    # get rid of potential NaNs
    xs_norm[np.isnan(xs_norm)] = -10
    ys_norm[np.isnan(ys_norm)] = -10

    # convert normalized coords to picture coords
    ys = (ys_norm * src.shape[0]).astype(np.float32)
    xs = (xs_norm * src.shape[1]).astype(np.float32)

    return xs, ys, bb
