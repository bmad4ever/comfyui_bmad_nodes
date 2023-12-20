import itertools

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


def remap_inner_cylinder(src, fov=90, swap_xy=False):
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


def remap_outer_cylinder(src, fov=90, swap_xy=False):
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

    m = f / omega;
    k = m + m ** -1;
    n = 1 - (1 / (1 + m ** -2))
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


def remap_inside_parabolas(src, roi_img, recalled=False):
    """
    @param roi_points_img: dst sized mask with the 6 points annotated
    @param recalled: safeguard against infinite recursive calls.
    @return: @return: x and y mappings ( to the original image pixels ) and roi bounding box coordinates
    """
    import skimage
    ret, roi_points_img = cv.threshold(roi_img, 127, 255, cv.THRESH_BINARY)  # ensure maximum value is 255
    skeleton = skimage.img_as_ubyte(skimage.morphology.skeletonize(roi_img))
    endpoints = find_endpoints(skeleton)

    # get contours and simplify them with poly
    contours, _ = cv.findContours(skeleton, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) != 2:
        raise (f"Parabola Remap requires exactly 2 drawn lines, however it obtained {len(contours)} contours.")

    # it is not a closed contour, but it should be easier to get the endpoints this way
    # the endpoints should be in the polyline to validate the input in the next step
    polylines = [cv.approxPolyDP(cnt, .005 * cv.arcLength(cnt, True), True)[:, 0] for cnt in contours]

    match validate_parabolas(polylines, endpoints):
        case 2:
            raise ("Couldn't match two endpoints to one of the contours."
                   "It may be the case that the parabolas are too flat.")
        case 1:  # current orientation won't work
            if not recalled:  # shouldn't get stuck, if equal proceeds; but will use a safeguard against potential oversights!
                xs, ys, bb = remap_inside_parabolas(cv.rotate(src, cv.ROTATE_90_CLOCKWISE)
                                                    , cv.flip(cv.rotate(roi_points_img, cv.ROTATE_90_CLOCKWISE), 1),
                                                    True)
                bb = (bb[1], bb[0], bb[3], bb[2])  # fix bb
                return np.rot90(np.fliplr(ys)), np.rot90(np.fliplr(xs)), bb
            # -- raise(...) -- no longer raise exception:
            # it will likely result in a messy map,
            # but may also work fine on some cases.
        case _:
            pass

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

    def alen_integral(x, a, b):
        return (((2 * a * x + b) ** 2 + 1) ** (1 / 2) * (2 * a * x + b) + np.arcsinh(2 * a * x + b)) / (4 * a)

    def arc_len(w, x1, x2):
        a_b_s = w[:, :, np.newaxis] * c1_coeffs[:2] + (1 - w[:, :, np.newaxis]) * c2_coeffs[:2]
        a = a_b_s[:, :, 0]
        b = a_b_s[:, :, 1]
        return alen_integral(x2, a, b) - alen_integral(x1, a, b)

    def compute_x_norm(ws, px):
        p0x, p0y = np.min(bottom_parabola, axis=0)
        p2x, p2y = np.max(bottom_parabola, axis=0)
        p3x, p3y = np.min(top_parabola, axis=0)
        p5x, p5y = np.max(top_parabola, axis=0)
        leftmost_xs = ws * p0x + (1 - ws) * p3x
        rightmost_xs = ws * p2x + (1 - ws) * p5x
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


# endregion remap parabolas


# region remap quadrilateral


def compute_homography(bottom_left_corner, bottom_right_corner, origin, src, upper_left_corner, upper_right_corner):
    src_pts = [[0, 0], [src.shape[1], 0], [0, src.shape[0]], [src.shape[1], src.shape[0]]]
    dst_pts = [upper_left_corner, upper_right_corner, bottom_left_corner, bottom_right_corner]
    src_pts, dst_pts = np.array(src_pts).astype(np.float32), np.array(dst_pts).astype(np.float32)
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
                raise ("invalid input; adjacent points with equal x!")  # this should never happen if input is valid!
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
    # b_x, b_y = upper_right_corner
    c_x, c_y = bottom_left_corner
    # d_x, d_y = bottom_right_corner
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


def remap_quadrilateral(src, roi_img, method):
    contours, hierarchy = cv.findContours(roi_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    moments = [cv.moments(c) for c in contours]
    centers = [(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in moments]

    bb = [
        min([c[0] for c in centers])
        , min([c[1] for c in centers])
        , max([c[0] for c in centers])
        , max([c[1] for c in centers])
    ]
    origin = [bb[0], bb[1]]
    bb_height = bb[3] - bb[1]
    bb_width = bb[2] - bb[0]

    # should already be ordered by y, but don't suppose this is the case
    centers.sort(key=lambda c: c[1])
    upper_left_corner = min(centers[:2], key=lambda c: c[0])
    upper_right_corner = max(centers[:2], key=lambda c: c[0])
    bottom_left_corner = min(centers[2:4], key=lambda c: c[0])
    bottom_right_corner = max(centers[2:4], key=lambda c: c[0])

    if method == "HOMOGRAPHY":
        h_matrix = compute_homography(bottom_left_corner, bottom_right_corner, origin, src, upper_left_corner,
                                      upper_right_corner)
        return h_matrix, bb

    xs = np.array([[x + origin[0] for x in range(bb_width)] for _ in range(bb_height)]).astype(np.float32)
    ys = np.array([[y + origin[1] for _ in range(bb_width)] for y in range(bb_height)]).astype(np.float32)

    xs_norm, ys_norm = quad_remap_methods_map[method](
        xs, ys, upper_left_corner, upper_right_corner, bottom_left_corner, bottom_right_corner)

    ys = ys_norm * src.shape[0]
    xs = xs_norm * src.shape[1]
    return xs, ys, bb

# endregion remap quadrilateral
