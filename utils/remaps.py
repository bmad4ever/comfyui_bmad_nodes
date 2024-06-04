import itertools

import numpy as np
import cv2 as cv
from numpy import ndarray


def xy_pixel_indexes(img, swap=False):
    height, width = img.shape[:2]
    xs = np.array([range(width)] * height).astype(np.float32)
    ys = np.array([[y for _ in range(width)] for y in range(height)]).astype(np.float32)

    if swap:
        return ys, xs, height, width
    else:
        return xs, ys, width, height


def quadratic_formulae(a, b, c):
    vs = np.sqrt(b ** 2 - 4 * a * c)
    a2 = a * 2
    return (-b - vs) / a2, (-b + vs) / a2


def remap_inner_cylinder(src: ndarray, fov=90, swap_xy=False):
    """
    Adapted from https://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection ;
    radius always equal to w; and f changes according to fov.

    @param fov: field of view in degrees
    @return: x and y mappings ( to the original image pixels )
    """
    xs, ys, w, h = xy_pixel_indexes(src, swap_xy)
    w_m1 = w - 1
    h_m1 = h - 1

    pc = [xs - w_m1 / 2, ys - h_m1 / 2]  # offset center

    ratio = np.tan(np.deg2rad(fov / 2))
    omega = w / 2
    f = omega / ratio
    r = w

    z0 = f - np.sqrt(r * r - omega * omega)
    _, zc = quadratic_formulae((pc[0] ** 2) / (f ** 2) + 1, -2 * z0, z0 ** 2 - r ** 2)

    final_point = [pc[0] * zc / f, pc[1] * zc / f]
    final_point[0] += w_m1 / 2
    final_point[1] += h_m1 / 2

    if swap_xy:
        return final_point[1], final_point[0], None
    else:
        return final_point[0], final_point[1], None


def remap_outer_cylinder(src: ndarray, fov=90, swap_xy=False):
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
    w_m1 = w - 1
    h_m1 = h - 1

    pc = [xs - w_m1 / 2, ys - h_m1 / 2]  # offset center

    ratio = np.tan(np.deg2rad(fov / 2))
    omega = w / 2
    f = omega / ratio

    m = f / omega
    k = m + m ** -1
    n = 1 - (1 / (1 + m ** -2))
    _, z0 = quadratic_formulae(a=1 - k ** -2 - n ** 2, b=-2 * f, c=f ** 2)
    r = z0 - f

    zc, _ = quadratic_formulae((pc[0] ** 2) / (f ** 2) + 1, -2 * z0, z0 ** 2 - r ** 2)
    final_point = [pc[0] * zc / f, pc[1] * zc / f]
    final_point[0] += w_m1 / 2
    final_point[1] += h_m1 / 2
    if swap_xy:
        return final_point[1], final_point[0], None
    else:
        return final_point[0], final_point[1], None


def remap_pinch_or_stretch(src: ndarray, power: tuple[float, float], center: tuple[float, float]):
    power_x, power_y = power
    px, py = center

    nsx, psx = px, 1 - px
    nsy, psy = py, 1 - py

    xs = [(x / (src.shape[1] - 1) - px) for _ in range(src.shape[0]) for x in range(src.shape[1])]
    ys = [(y / (src.shape[0] - 1) - py) for y in range(src.shape[0]) for _ in range(src.shape[1])]
    xs = np.array(xs).astype(np.float32).reshape(src.shape[:2])
    ys = np.array(ys).astype(np.float32).reshape(src.shape[:2])
    xs[xs < 0] /= nsx
    xs[xs > 0] /= psx
    ys[ys < 0] /= nsy
    ys[ys > 0] /= psy

    xs = np.sign(xs) * (1 - np.power(1 - np.abs(xs), power_x))
    ys = np.sign(ys) * (1 - np.power(1 - np.abs(ys), power_y))
    xs[xs < 0] *= nsx
    xs[xs > 0] *= psx
    ys[ys < 0] *= nsy
    ys[ys > 0] *= psy
    xs += px
    ys += py
    xs *= (src.shape[1] - 1)
    ys *= (src.shape[0] - 1)

    return xs, ys, None


def lens_undistort(r: ndarray | float, a: float, b: float, c: float, d: float | None = None):
    if d is None:
        d = 1 - (a + b + c)
    return (a * r ** 3 + b * r ** 2 + c * r + d) * r


def lens_undistort_inv(r: ndarray | float, a: float, b: float, c: float, d: float | None = None):
    a, b, c = -a, -b, -c
    if d is None:
        d = 1 - (a + b + c)
    return r / (a * r ** 3 + b * r ** 2 + c * r + d)


def remap_barrel_distortion(src: ndarray, a: float, b: float, c: float, d: float | None, inverse: bool):
    """
    Similar to magick's barrel distort.
    Can be used to: undistort images from camera/lens combo; create barrel, pincushion or mustache distortion.

    inverse. -> use alternative formula to compute big R  ( not mathematical inverse func of the vanilla func )
    """
    xs = [x - (src.shape[1] - 1) / 2 for _ in range(src.shape[0]) for x in range(src.shape[1])]
    ys = [y - (src.shape[0] - 1) / 2 for y in range(src.shape[0]) for _ in range(src.shape[1])]
    xs = np.array(xs).astype(np.float32).reshape(src.shape[:2])
    ys = np.array(ys).astype(np.float32).reshape(src.shape[:2])

    radii, radians = cv.cartToPolar(xs, ys, angleInDegrees=False)
    min_whr = min(radii[0, src.shape[1] // 2], radii[src.shape[0] // 2, 0])
    norm_radii = radii / min_whr

    undistort_method = lens_undistort_inv if inverse else lens_undistort
    new_radii = undistort_method(norm_radii, a, b, c, d)

    new_radii *= min_whr
    xs, ys = cv.polarToCart(new_radii, radians)
    xs += src.shape[1] / 2
    ys += src.shape[0] / 2
    return xs, ys, None


def remap_reverse_barrel_distortion(src: ndarray, a: float, b: float, c: float, d: float | None, inverse: bool):
    from numpy.polynomial import Polynomial
    from joblib import Parallel, delayed

    if d is None:
        d = 1 - (a + b + c)

    xs = [x - (src.shape[1] - 1) / 2 for _ in range(src.shape[0]) for x in range(src.shape[1])]
    ys = [y - (src.shape[0] - 1) / 2 for y in range(src.shape[0]) for _ in range(src.shape[1])]
    xs = np.array(xs).astype(np.float32).reshape(src.shape[:2])
    ys = np.array(ys).astype(np.float32).reshape(src.shape[:2])

    radii, radians = cv.cartToPolar(xs, ys, angleInDegrees=False)
    min_whr = min(radii[0, src.shape[1] // 2], radii[src.shape[0] // 2, 0])
    rs = radii / min_whr

    def compute_roots(poly, j, i):
        return next(
            filter(
                lambda e: 2 >= e.real >= 0 and abs(e.imag) < 1e-8,
                (poly - rs[j, i]).roots()
            )
            , -10 + 0j).real

    if inverse:
        # if using inverse barrel formulae

        # don't know how to solve this; but an approximation seems to suffice.
        # not sure if this is rly smart or rly dum; is there a simpler solution? I might be using a tank to kill a fly
        samples = 512
        max_rs = np.max(rs)
        radii_range = int(max_rs * samples)
        samples = float(samples)
        small_rs = np.array([(float(x)) / samples for x in range(radii_range)]).astype(np.float32)

        # try to guess whether it is barrel or pincushion
        reference_r = 1.05
        ref_dist_r = lens_undistort_inv(reference_r, a, b, c)
        expanded = ref_dist_r < reference_r

        def find_scaled_max_rs(fn):
            from scipy.optimize import dual_annealing
            ret = dual_annealing(fn, bounds=list(zip([0], [2])))
            return ret.x

        mrs = find_scaled_max_rs(lambda r: abs(max_rs - lens_undistort_inv(r, a, b, c)))
        small_rs *= mrs / max_rs
        big_rs = lens_undistort_inv(small_rs, a, b, c)

        if expanded:
            # has expanded -> polyfit in reverse and use solve over the poly to readjust
            poly = np.polyfit(small_rs, big_rs, 7)[::-1]  # -> Polynomial receives coeffs in reverse order
            poly = Polynomial(poly)
            rev_radii = Parallel(n_jobs=-1)(delayed(compute_roots)(poly, j, i)
                                            for j in range(rs.shape[0]) for i in range(rs.shape[1]))
            rev_radii = np.array(rev_radii).astype(np.float32).reshape(rs.shape)
        else:
            # otherwise -> polyfit and use poly directly to readjust
            poly = np.polyfit(big_rs, small_rs, 7)
            poly = np.poly1d(poly)
            rev_radii = poly(rs)
    else:
        # if using normal barrel formulae
        poly = Polynomial([0, d, c, b, a])
        rev_radii = Parallel(n_jobs=-1)(delayed(compute_roots)(poly, j, i)
                                        for j in range(rs.shape[0]) for i in range(rs.shape[1]))
        rev_radii = np.array(rev_radii).astype(np.float32).reshape(rs.shape)

    rev_radii *= min_whr
    xs, ys = cv.polarToCart(rev_radii, radians)
    xs += src.shape[1] // 2
    ys += src.shape[0] // 2
    return xs, ys, None


# region remap parabolas

def get_quadratic_curve_coeffs(p0, p1, p2):
    """
    DEPRECATED -> currently replaced with np.polyfit using additional points
    Estimate quadratic curve coefficients from the given points: p0, p1, p2
    @return: coefficienst: [ a, b, c ]
    """
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2

    a = np.array([[x1 ** 2, x1, 1], [x2 ** 2, x2, 1], [x3 ** 2, x3, 1]])
    b = np.array([y1, y2, y3])
    return np.linalg.solve(a, b)


def find_endpoints(skeleton):
    """
    Finds the edges of the morphological skeleton.
    Won't work for lines ticker than 1 pixel.
    @param skeleton: binary image with values of 0 and 255 that contains the morphological skeleton
    """
    kernel = np.array(
        [[1, 1, 1],
         [1, 10, 1],
         [1, 1, 1]])
    skeleton = cv.filter2D(skeleton / 255, -1, kernel)
    indices = np.nonzero(skeleton == 11)
    return list(zip(indices[1], indices[0]))


def validate_parabolas(lines, endpoints):
    """
    @return: integer code:
        0: okay, the data seems valid
        1: data might work using other orientation
        2: data won't work, some lines did not match at least two endpoints
    """
    LEEWAY = 8 ** 2  # given as the squared number of pixels
    for cnt in lines:
        cnt_endpoints = [point for point in endpoints if any(np.sum((point - cnt) ** 2, 1) <= LEEWAY)]

        # remove endpoints from list, so that there are less checks in the next line
        endpoints = [point for point in endpoints if point not in cnt_endpoints]

        left_endpoint, right_endpoint = sorted(cnt_endpoints, key=lambda p: p[0])

        if len(cnt_endpoints) < 2:
            return 2  # not enough points, won't work with or without rotation

        if left_endpoint == right_endpoint:
            return 1  # if they have the same x there is no need to check other points in the next check

        if any(not (left_endpoint[0] < point[0] < right_endpoint[0]) for point in cnt if
               tuple(point) not in cnt_endpoints):
            return 1  # can't be defined as a function, but may work in another orientation
    return 0  # looks fine


def get_knee_point(endpoints, line_points):
    """
    @param endpoints: x sorted (increasing) endpoints of the given line.
    @param line_points: ...
    @return: the knee point of the given line
    """

    def angle_at_p(e1, p, e2):
        """
        angle between vector starting at point "p" and ending at "e1" (first endpoint) and
                vector starting at point "p" and ending at "e2" (second endpoint)
        """
        vector1 = e1 - p
        vector2 = e2 - p
        dot_product = np.dot(vector1, vector2)
        magnitude_vector1 = np.linalg.norm(vector1)
        magnitude_vector2 = np.linalg.norm(vector2)
        LEEWAY = 8  # consider points at this distance the same
        if magnitude_vector2 <= LEEWAY or magnitude_vector1 <= LEEWAY:
            # point roughly coincides with one of the endpoints, ignore it as a potential knee candidate
            return float("inf")
        cosine_angle = dot_product / (magnitude_vector1 * magnitude_vector2)
        angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return angle_radians

    return min(line_points, key=lambda p: angle_at_p(endpoints[0], p, endpoints[1]))


def alen_integral(x, a, b):
    """
    arc length primitive for the parabola(s) given by :  a*x**2 + b*x + c ;  at the given point(s) x.
    """
    return (((2 * a * x + b) ** 2 + 1) ** (1 / 2) * (2 * a * x + b) + np.arcsinh(2 * a * x + b)) / (4 * a)


def get_parabolas_edges_xs(ws, bottom_parabola, top_parabola):
    """
    get left and right edges of all parabolas segments between bottom and top parabolas' segments
    """
    p0x, _ = np.min(bottom_parabola, axis=0)
    p2x, _ = np.max(bottom_parabola, axis=0)
    p3x, _ = np.min(top_parabola, axis=0)
    p5x, _ = np.max(top_parabola, axis=0)
    leftmost_xs = ws * p0x + (1 - ws) * p3x
    rightmost_xs = ws * p2x + (1 - ws) * p5x
    return leftmost_xs, rightmost_xs


def remap_inside_parabolas(src, roi_img, recalled=False):
    """
    Generic implementation applied to both simple and advanced parabola remaps.
    @param roi_img: dst sized mask with the 6 points annotated
    @param recalled: safeguard against infinite recursive calls.
    @return: @return: x and y mappings ( to the original image pixels ) and roi bounding box coordinates
    """
    endpoints, polylines, roi_points_img = get_parabolas_pair(roi_img)

    match validate_parabolas(polylines, endpoints):
        case 2:
            raise Exception("Couldn't match two endpoints to one of the contours."
                            "It may be the case that the parabolas are too flat.")
        case 1:  # current orientation won't work
            if not recalled:  # shouldn't get stuck, if equal proceeds; but will use a safeguard against potential oversights!
                xs, ys, bb, _ = remap_inside_parabolas(cv.rotate(src, cv.ROTATE_90_CLOCKWISE)
                                                       , cv.flip(cv.rotate(roi_points_img, cv.ROTATE_90_CLOCKWISE), 1),
                                                       True)
                bb = (bb[1], bb[0], bb[3], bb[2])  # fix bb
                return np.rot90(np.fliplr(ys)), np.rot90(np.fliplr(xs)), bb, True
            # -- raise(...) -- no longer raise exception:
            # it will likely result in a messy map,
            # but may also work fine on some cases.

    # get parabolas by Y coordinate
    top_parabola, bottom_parabola = sorted(polylines, key=lambda pl: max(p[1] for p in pl))

    # get upper and lower bb leeway
    lower_limit_leeway = max(bottom_parabola, key=lambda c: c[1])[1] - min(bottom_parabola, key=lambda c: c[1])[1]
    upper_limit_leeway = max(top_parabola, key=lambda c: c[1])[1] - min(top_parabola, key=lambda c: c[1])[1]

    def all_points():
        return itertools.chain(polylines[0], polylines[1])

    bb = [
        int(min([p[0] for p in all_points()]))
        , int(min([p[1] for p in all_points()]) - upper_limit_leeway)
        , int(max([p[0] for p in all_points()]))
        , int(max([p[1] for p in all_points()]) + lower_limit_leeway)
    ]
    # clip lower and upper bounds (within dst)
    bb[1] = max(bb[1], 0)
    bb[3] = min(bb[3], roi_points_img.shape[0])

    origin = [bb[0], bb[1]]
    dst_box_height = int(bb[3] - bb[1])
    dst_box_width = int(bb[2] - bb[0])

    # get the quadratic curves coefficients from the parsed points
    c1_coeffs = np.polyfit(bottom_parabola[:, 0], bottom_parabola[:, 1], 2)
    c2_coeffs = np.polyfit(top_parabola[:, 0], top_parabola[:, 1], 2)

    # region auxiliary functions definitions
    def compute_w(x, y):
        """
        y = (w*a1 + w*(a2-w))x^2 + (w*b1 + w*(b2-w))x + (w*c1 + w*(c2-w)
        rearranged to compute w
        """
        a_1, b_1, c_1 = c1_coeffs
        a_2, b_2, c_2 = c2_coeffs
        return -(y - a_2 * x ** 2 - b_2 * x - c_2) / ((a_2 - a_1) * x ** 2 + (b_2 - b_1) * x + c_2 - c_1)

    def arc_len(w, x1, x2):
        a_b_s = w[:, :, np.newaxis] * c1_coeffs[:2] + (1 - w[:, :, np.newaxis]) * c2_coeffs[:2]
        a = a_b_s[:, :, 0]
        b = a_b_s[:, :, 1]
        return alen_integral(x2, a, b) - alen_integral(x1, a, b)

    def compute_x_norm(ws, px):
        leftmost_xs, rightmost_xs = get_parabolas_edges_xs(ws, bottom_parabola, top_parabola)
        full_segs_lens = arc_len(ws, leftmost_xs, rightmost_xs)
        to_point_lens = arc_len(ws, leftmost_xs, px)

        xsn = to_point_lens / full_segs_lens

        # clip sections outside roi
        xsn[to_point_lens > full_segs_lens] = -10
        xsn[to_point_lens < 0] = -10

        return xsn

    # endregion

    # compute ys
    ys_norm = np.array([[compute_w(x + origin[0], y + origin[1]) for x in range(dst_box_width)] for y in
                        range(dst_box_height)]).astype(np.float32)
    # clip sections outside roi
    ys_norm[ys_norm > 1] = -10
    ys_norm[ys_norm < 0] = -10

    # compute xs
    xs_pxs = np.array([[x + origin[0] for x in range(dst_box_width)] for _ in range(dst_box_height)]).astype(np.float32)
    xs_norm = compute_x_norm(ys_norm, xs_pxs)
    # outside roi clip is already done by compute_x_norm in previous step

    # get rid of potential NaNs
    xs_norm[np.isnan(xs_norm)] = -10
    ys_norm[np.isnan(ys_norm)] = -10

    return xs_norm, ys_norm, bb, recalled


def get_parabolas_pair(roi_img):
    import skimage
    _, roi_points_img = cv.threshold(roi_img, 127, 255, cv.THRESH_BINARY)
    skeleton = skimage.img_as_ubyte(skimage.morphology.skeletonize(roi_img))
    endpoints = find_endpoints(skeleton)

    contours, _ = cv.findContours(skeleton, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) != 2:
        raise ValueError(f"Parabola Remap requires exactly 2 drawn lines,"
                         f" however it obtained {len(contours)} contours.")

    # it is not a closed contour, but it should be easier to get the endpoints this way
    # the endpoints should be in the polyline to validate the input in the next step
    polylines = [cv.approxPolyDP(cnt, .005 * cv.arcLength(cnt, True), True)[:, 0] for cnt in contours]

    return endpoints, polylines, roi_points_img


def remap_inside_parabolas_simple(src, roi_img):
    xs_norm, ys_norm, bb, _ = remap_inside_parabolas(src, roi_img)

    # convert normalized coords to picture coords
    ys = (ys_norm * (src.shape[0] - 1)).astype(np.float32)
    xs = (xs_norm * (src.shape[1] - 1)).astype(np.float32)
    return xs, ys, bb


def remap_inside_parabolas_advanced(src, roi_img, curve_adjust, ortho_adjust, flip_ortho_adj):
    xs_norm, ys_norm, bb, swap = remap_inside_parabolas(src, roi_img)

    def curve_wise_adjust(ss_norm):
        ss_outbounds_indices = (ss_norm == -10)
        ss_norm = ss_norm.astype(np.float32)
        ss_norm = (ss_norm - .5) * 2
        xs_sign = np.sign(ss_norm)
        ss_norm = (1 - np.abs(ss_norm)) * xs_sign
        ss_norm = np.abs(np.float_power(ss_norm, curve_adjust, dtype=complex)) * xs_sign
        ss_norm = (1 - np.abs(ss_norm)) * xs_sign
        ss_norm = ss_norm * .5 + .5
        ss_norm[ss_outbounds_indices] = -10  # jic, make sure no bleeding occurs
        return ss_norm

    def ortho_wise_adjust(ss_norm):
        ss_outbounds_indices = (ss_norm == -10)
        if flip_ortho_adj:
            ss_norm = 1 - ss_norm
        ss_norm = np.abs(np.float_power(ss_norm, ortho_adjust, dtype=complex)) * np.sign(ss_norm)
        if flip_ortho_adj:
            ss_norm = 1 - ss_norm
        ss_norm[ss_outbounds_indices] = -10  # jic, make sure no bleeding occurs
        return ss_norm

    if swap:
        ys_norm = curve_wise_adjust(ys_norm)
        xs_norm = ortho_wise_adjust(xs_norm)
    else:
        xs_norm = curve_wise_adjust(xs_norm)
        ys_norm = ortho_wise_adjust(ys_norm)

    # convert normalized coords to picture coords
    ys = (ys_norm * (src.shape[0] - 1)).astype(np.float32)
    xs = (xs_norm * (src.shape[1] - 1)).astype(np.float32)
    return xs, ys, bb


def remap_from_inside_parabolas(_, roi_img, dst_width: int, dst_height: int, recalled=False):
    def approx_rev_alen_integral(r, a, b, x_s, x_f):
        from numpy.polynomial import Polynomial as pol
        from joblib import Parallel, delayed

        x_interval = x_f - x_s
        n_points = 5
        xx = np.array([x_s + i * x_interval / n_points for i in range(n_points)])
        ll = alen_integral(xx, a, b)

        def compute_roots(j, i):
            return next(
                filter(
                    lambda e: x_f[j, i] >= e.real >= x_s[j, i] and abs(e.imag) < 1e-5,
                    (pol.fit(x=xx[:, j, i], y=ll[:, j, i], deg=3) - r[j, i]).roots()
                )
                , -10 + 0j).real

        x_s = Parallel(n_jobs=-1)(delayed(compute_roots)(j, i) for j in range(a.shape[0]) for i in range(a.shape[1]))
        return np.array(x_s).astype(np.float32).reshape(a.shape)

    def compute_x(a, b, x_s, x_f, x):
        # x / w = ix_len / t_len <=> x/w * t_len = ix_len <=> ... ( replace lens w/ alen_integrals )
        x_s_l = alen_integral(x_s, a, b)
        return approx_rev_alen_integral(
            x / dst_width * (alen_integral(x_f, a, b) - x_s_l) + x_s_l,
            # alen_integral(x_s, a, b) + x,
            a, b, x_s, x_f
        )

    def compute_ys(xs, a, b, c):  # the ez part, lol
        return a * np.power(xs, 2) + b * xs + c

    # get points for parabola
    endpoints, polylines, roi_points_img = get_parabolas_pair(roi_img)

    match validate_parabolas(polylines, endpoints):
        case 2:
            raise Exception("Couldn't match two endpoints to one of the contours."
                            "It may be the case that the parabolas are too flat.")
        case 1:  # current orientation won't work
            if not recalled:  # shouldn't get stuck, if equal proceeds; but will use a safeguard against potential oversights!
                xs, ys, _ = remap_from_inside_parabolas(None,
                                                        cv.flip(cv.rotate(roi_points_img, cv.ROTATE_90_CLOCKWISE), 1),
                                                        dst_width, dst_height, True)
                return np.rot90(np.fliplr(ys)), np.rot90(np.fliplr(xs)), None
            # -- raise(...) -- no longer raise exception:
            # it will likely result in a messy map,
            # but may also work fine on some cases.

    # get parabolas by Y coordinate
    top_parabola, bottom_parabola = sorted(polylines, key=lambda pl: max(p[1] for p in pl))

    # get parabolas coefficients
    c1_coeffs = np.polyfit(bottom_parabola[:, 0], bottom_parabola[:, 1], 2)
    c2_coeffs = np.polyfit(top_parabola[:, 0], top_parabola[:, 1], 2)

    # get all points' parabolas coefficients
    h_m1 = dst_height - 1
    ws = np.array([j / h_m1 for j in range(dst_height) for _ in range(dst_width)]).reshape(dst_height, dst_width)
    a_b_c_s = ws[:, :, np.newaxis] * c1_coeffs + (1 - ws[:, :, np.newaxis]) * c2_coeffs
    a = a_b_c_s[:, :, 0]
    b = a_b_c_s[:, :, 1]
    c = a_b_c_s[:, :, 2]

    # get edges in src and xs in dst
    leftmost_xs, rightmost_xs = get_parabolas_edges_xs(ws, bottom_parabola, top_parabola)
    x = np.array([i for _ in range(dst_height) for i in range(dst_width)]).reshape(dst_height, dst_width)

    # compute xs & ys
    xs = compute_x(a, b, leftmost_xs, rightmost_xs, x).astype(np.float32)
    ys = compute_ys(xs, a, b, c).astype(np.float32)
    return xs, ys, None


# endregion remap parabolas


# region remap quadrilateral


def compute_homography(
        src, origin, sorted_dst_pts
):
    """
    @param src: source image ( to apply the transformation to )
    @param origin: dst quad bounding box top-left corner
    @param sorted_dst_pts: the dst quad points sorted from left to right and top to bottom
    @return: homography matrix
    """
    src_pts = [[0, 0], [src.shape[1], 0], [0, src.shape[0]], [src.shape[1], src.shape[0]]]
    src_pts, dst_pts = np.array(src_pts).astype(np.float32), np.array(sorted_dst_pts).astype(np.float32)
    dst_pts -= origin
    h_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    return h_matrix


def remap_quadrilateral_edge_pairs_interpolation(
        xs, ys,
        upper_left_corner, upper_right_corner,
        bottom_left_corner, bottom_right_corner,
):
    """
    Imagine a line that is the weighted average of 2 edges, x-wise, or y-wise.
    The weight is x / width ( or y / height ).

    If no edge is parallel these lines follow 2 different vanishing points,
     but the mapped image doesn't shrink according to these!

    A bit funky; Not sure if it could be of use... but it is easy to implement.
    """

    def compute_m(pt1, pt2, limit_m=None):
        """
        Slope of the line
        @param pt1: a point on the line
        @param pt2: another point on the line
        @param limit_m: x's are equal, return this instead of m; if not set raise an exception.
        @return:
        """
        if pt1[0] == pt2[0]:
            if limit_m is None:
                # this should never happen if input is valid!
                raise ValueError("invalid input; adjacent points with equal x!")
            else:
                return limit_m

        pt1_to_pt2 = np.array(pt2) - np.array(pt1)
        return pt1_to_pt2[1] / pt1_to_pt2[0]

    def compute_b(pt, m):
        """
        @param pt: a point on the line
        @param m: slope of the line
        @return: Y value at x = 0
        """
        return pt[1] - m * pt[0]

    m12 = compute_m(upper_left_corner, upper_right_corner)
    m34 = compute_m(bottom_left_corner, bottom_right_corner)
    b12 = compute_b(upper_left_corner, m12)
    b34 = compute_b(bottom_left_corner, m34)

    def compute_norm_ys(xs, ys):
        ws = (ys - m12 * xs - b12) / ((m34 - m12) * xs + b34 - b12)
        ws[(ws > 1) | (ws < 0) | (np.isnan(ws))] = -10  # clear bounds
        return ws

    def compute_xs_norm(xs, ws, ys):
        bb_height2 = ys.shape[0] ** 2
        m13 = compute_m(upper_left_corner, bottom_left_corner, bb_height2)
        m24 = compute_m(upper_right_corner, bottom_right_corner, bb_height2)
        b13 = compute_b(upper_left_corner, m13)
        b24 = compute_b(upper_right_corner, m24)

        # why not just compute ws for X?
        # should work exactly the same

        xs_os = -((b34 - b12) * ws - b13 + b12) / ((m34 - m12) * ws - m13 + m12)
        xs_es = -((b34 - b12) * ws - b24 + b12) / ((m34 - m12) * ws - m24 + m12)
        ys_os = m13 * xs_os + b13
        ys_es = m24 * xs_es + b24

        xs_lens = np.sqrt((ys_es - ys_os) ** 2 + (xs_es - xs_os) ** 2)
        xs_minus_xs_os = xs - xs_os
        xs_norms = np.sign(xs_minus_xs_os) * np.sqrt((ys - ys_os) ** 2 + xs_minus_xs_os ** 2) / xs_lens
        # notice that np.sign is used to address out of bounds coordinates
        xs_norms[(xs_norms > 1) | (xs_norms < 0) | (np.isnan(xs_norms))] = -10  # clear bounds
        return xs_norms

    ys_norm = compute_norm_ys(xs, ys)
    xs_norm = compute_xs_norm(xs, ys_norm, ys)

    return xs_norm, ys_norm


def remap_quadrilateral_lengthwise(
        xs, ys,
        upper_left_corner, upper_right_corner,
        bottom_left_corner, bottom_right_corner,
):
    """
    Imagine a line that connects 2 points at 2 opposite edges, at the same % of the length at each edge.
    That % of length is : x / width ( or y / height )
    [ implementation might be wrong, but it does appear correct in first tests ]
    """

    a_x, a_y = upper_left_corner
    c_x, c_y = bottom_left_corner
    ab = (np.array(upper_right_corner) - np.array(upper_left_corner)).astype(np.float32)
    cd = (np.array(bottom_right_corner) - np.array(bottom_left_corner)).astype(np.float32)
    ab_x, ab_y = ab
    cd_x, cd_y = cd

    def compute_xs_norm(xs, ys):
        # This is UGLY AF, my eyes burn!!!
        #   Too tired to think about it;
        #   so, I just copy-pasted the result from eq. system. Fingers crossed, lol ...
        #  Can't recall if this is length-wise;
        #   judging by the result, and since it contains a sqrt operation it seems to be...
        #  TODO, prob a good idea to review this if I get the time... and simplify if possible...
        return (np.sqrt(
            (cd_x ** 2 - 2 * ab_x * cd_x + ab_x ** 2) * ys ** 2 + (
                    ((2 * ab_x - 2 * cd_x) * cd_y + 2 * ab_y * cd_x - 2 * ab_x * ab_y) * xs +
                    (2 * a_x * cd_x - 4 * ab_x * c_x + 2 * a_x * ab_x) * cd_y - 2 * a_y * cd_x ** 2 + (
                            2 * ab_x * c_y + 2 * ab_y * c_x - 4 * a_x * ab_y + 2 * a_y * ab_x) * cd_x
                    - 2 * ab_x ** 2 * c_y + 2 * ab_x * ab_y * c_x) * ys + (
                    cd_y ** 2 - 2 * ab_y * cd_y + ab_y ** 2) * xs ** 2
            + (-2 * a_x * cd_y ** 2 + (
                    2 * a_y * cd_x + 2 * ab_x * c_y + 2 * ab_y * c_x + 2 * a_x * ab_y - 4 * a_y * ab_x) * cd_y + (
                       2 * a_y * ab_y - 4 * ab_y * c_y) * cd_x
               + 2 * ab_x * ab_y * c_y - 2 * ab_y ** 2 * c_x) * xs + a_x ** 2 * cd_y ** 2 + (
                    -2 * a_x * a_y * cd_x - 2 * a_x * ab_x * c_y +
                    (4 * a_y * ab_x - 2 * a_x * ab_y) * c_x) * cd_y + a_y ** 2 * cd_x ** 2 + (
                    (4 * a_x * ab_y - 2 * a_y * ab_x) * c_y - 2 * a_y * ab_y * c_x) * cd_x +
            ab_x ** 2 * c_y ** 2 - 2 * ab_x * ab_y * c_x * c_y + ab_y ** 2 * c_x ** 2
        ) + (ab_x - cd_x) * ys + (cd_y - ab_y) * xs - a_x * cd_y + a_y * cd_x - ab_x * c_y + ab_y * c_x) / (
                2 * ab_x * cd_y - 2 * ab_y * cd_x)

    def compute_ys_norm(ys, xs, px, a, ab, c, cd):
        px = px[:, :, np.newaxis]
        pa = a + px * ab
        pc = c + px * cd

        ly = ys - pa[:, :, 1]
        lx = xs - pa[:, :, 0]

        pp_lengths = np.linalg.norm(pa - pc, axis=2)
        pl_lengths = np.sqrt(ly ** 2 + lx ** 2)

        pl_lengths[ly <= 0] *= -1  # preserve sign in order to detect out of bounds
        return pl_lengths / pp_lengths  # ys_norm

    xs_norm = compute_xs_norm(xs, ys)
    xs_norm[(xs_norm > 1) | (xs_norm < 0) | (np.isnan(xs_norm))] = -10  # clear out of bounds

    ys_norm = compute_ys_norm(ys, xs, xs_norm, upper_left_corner, ab, bottom_left_corner, cd)
    ys_norm[(ys_norm > 1) | (ys_norm < 0) | (np.isnan(ys_norm))] = -10  # clear out of bounds

    return xs_norm, ys_norm.astype(np.float32)  # ?


quad_remap_methods_map = {
    "HOMOGRAPHY": None,  # does not compute xs, ys; only the homography matrix
    "LENGTH-WISE": remap_quadrilateral_lengthwise,
    "W-EDGE-PAIR": remap_quadrilateral_edge_pairs_interpolation
}


def remap_quadrilateral(src: ndarray, roi_img: ndarray, method: str) -> tuple[ndarray, ndarray, list[int]] | tuple[
    ndarray, list[int]]:
    quad_corners = get_quad_corners(roi_img)
    bb, bb_width, bb_height, origin = get_quad_bounding_box(quad_corners)
    quad_corners = get_ordered_corners(quad_corners)

    if method == "HOMOGRAPHY":
        h_matrix = compute_homography(src, origin, quad_corners)
        return h_matrix, bb

    xs = np.array([[x + origin[0] for x in range(bb_width)] for _ in range(bb_height)]).astype(np.float32)
    ys = np.array([[y + origin[1] for _ in range(bb_width)] for y in range(bb_height)]).astype(np.float32)

    xs_norm, ys_norm = quad_remap_methods_map[method](xs, ys, *quad_corners)

    ys = ys_norm * (src.shape[0] - 1)
    xs = xs_norm * (src.shape[1] - 1)
    return xs, ys, bb


def get_quad_bounding_box(quad_corners: list[tuple[int, int]]) -> tuple[list[int], int, int, tuple[int, int]]:
    """
    @param quad_corners: unordered quad corners
    @return: bounding box (bb) corners, bb width, bb height, origin ( bb top-left corner )
    """
    bb = [
        min([c[0] for c in quad_corners])
        , min([c[1] for c in quad_corners])
        , max([c[0] for c in quad_corners])
        , max([c[1] for c in quad_corners])
    ]
    bb_width = bb[2] - bb[0]
    origin = (bb[0], bb[1])
    bb_height = bb[3] - bb[1]

    return bb, bb_width, bb_height, origin


def get_ordered_corners(quad_corners: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
        note: sorts original input list by y
        @return: from top to bottom and left to right, similar to text
    """
    # should already be ordered by y, but don't suppose this is the case
    quad_corners.sort(key=lambda c: c[1])
    upper_left_corner = min(quad_corners[:2], key=lambda c: c[0])
    upper_right_corner = max(quad_corners[:2], key=lambda c: c[0])
    bottom_left_corner = min(quad_corners[2:4], key=lambda c: c[0])
    bottom_right_corner = max(quad_corners[2:4], key=lambda c: c[0])
    return [upper_left_corner, upper_right_corner, bottom_left_corner, bottom_right_corner]


def get_quad_corners(roi_img) -> list[tuple[int, int]]:
    contours, _ = cv.findContours(roi_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    moments = [cv.moments(c) for c in contours]
    centers = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in moments]
    return centers


def remap_from_quadrilateral(_, roi_img: ndarray, dst_width: int, dst_height: int) -> tuple[ndarray, list[int]]:
    # ONLY IMPLEMENTED FOR HOMOGRAPHY

    quad_corners = get_quad_corners(roi_img)
    quad_corners = get_ordered_corners(quad_corners)

    dst_pts = [[0, 0], [dst_width, 0], [0, dst_height], [dst_width, dst_height]]
    dst_bb = [0, 0, dst_width, dst_height]

    src_pts, dst_pts = np.array(quad_corners).astype(np.float32), np.array(dst_pts).astype(np.float32)
    h_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
    return h_matrix, dst_bb

# endregion remap quadrilateral
