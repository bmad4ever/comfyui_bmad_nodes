import numpy as np
from cupyx import jit


@jit.rawkernel()
def blur_cuda(src, mask, out, kernel_s, w, h):
    tidx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    tidy = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y

    kernel_r2 = (kernel_s / 2) ** 2
    ntidx = jit.gridDim.x * jit.blockDim.x
    ntidy = jit.gridDim.y * jit.blockDim.y

    for j in range(tidy, h, ntidy):
        ymin = max(j - kernel_s // 2, 0)
        ymax = min(j + kernel_s // 2, h - 1)
        for i in range(tidx, w, ntidx):
            if mask[j, i] == 0.0:
                xmin = max(i - kernel_s // 2, 0)
                xmax = min(i + kernel_s // 2, w - 1)
                count = 0.0
                avg_c0 = 0.0
                avg_c1 = 0.0
                avg_c2 = 0.0
                for y in range(ymin, ymax):
                    for x in range(xmin, xmax):
                        if mask[y, x] > 0.0:
                            dist2 = (j - y) ** 2 + (i - x) ** 2
                            if dist2 <= kernel_r2:
                                count += 1.0
                                avg_c0 = avg_c0 * (count - 1) / count + src[y, x, 0] / count
                                avg_c1 = avg_c1 * (count - 1) / count + src[y, x, 1] / count
                                avg_c2 = avg_c2 * (count - 1) / count + src[y, x, 2] / count
                out[j, i, 0] = avg_c0
                out[j, i, 1] = avg_c1
                out[j, i, 2] = avg_c2


def blur_cpu(src, mask, kernel_s, w, h):
    """
    IMPORTANT DEV-NOTE: unlike GPU solution, src image corners are skipped;
    this way relevant kernel indexes are precomputed
    and there are no distance computations within the inner loop
    """
    import multiprocessing
    from joblib import Parallel, delayed
    # this is safe, right? ( --___-- );
    # high kernel size can take forever to compute & idk how to abort it midway yet;
    # prob. a good idea to avoid using all cores.

    # region auxiliary methods

    def precompute_kernel_indices(kernel_r):
        y, x = np.ogrid[-kernel_r:kernel_r + 1, -kernel_r:kernel_r + 1]
        mask = x ** 2 + y ** 2 <= kernel_r ** 2
        indices = np.array(np.where(mask)).T
        return indices - [kernel_r, kernel_r]

    def calculate_average_color(src, mask, kernel_indices, j, i):
        valid_indices = kernel_indices + [j, i]
        valid_pixels = mask[valid_indices[:, 0], valid_indices[:, 1]] > 0.0
        if valid_pixels.any():
            avg_c = np.mean(src[valid_indices[valid_pixels, 0], valid_indices[valid_pixels, 1]], axis=0)
        else:
            avg_c = np.zeros(3)
        return avg_c

    def process_chunk(args):
        src, mask, kernel_s, w, h, start_row, end_row = args
        kernel_r = kernel_s // 2
        kernel_indices = precompute_kernel_indices(kernel_r)
        out_chunk = np.zeros((end_row - start_row, w, src.shape[2]))

        for j in range(max(start_row, kernel_r), min(end_row, h - kernel_r)):
            for i in range(kernel_r, w - kernel_r):
                if mask[j, i] > 0.0:
                    continue
                avg_c = calculate_average_color(src, mask, kernel_indices, j, i)
                out_chunk[j - start_row, i] = avg_c
        return out_chunk

    # endregion

    # setup args
    num_cores = multiprocessing.cpu_count()
    num_cores = max(1, num_cores-1)  # leave 1 core unoccupied in case computations take forever
    chunk_size = h // num_cores
    chunks = [(src, mask, kernel_s, w, h, i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    # include any leftover rows
    if h % num_cores != 0:
        chunks[-1] = (src, mask, kernel_s, w, h, (num_cores - 1) * chunk_size, h)

    # compute using available cores
    with Parallel(n_jobs=num_cores) as parallel:
        out_chunks = parallel(delayed(process_chunk)(chunk) for chunk in chunks)

    out = np.vstack(out_chunks)   # combine results

    return out
