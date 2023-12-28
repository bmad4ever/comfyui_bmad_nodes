import numpy as np
from cupyx import jit


@jit.rawkernel()
def blur_cuda(src, mask, out, kernel, kernel_s, w, h):
    tidx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    tidy = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y

    kernel_r = kernel_s // 2
    ntidx = jit.gridDim.x * jit.blockDim.x
    ntidy = jit.gridDim.y * jit.blockDim.y
    for j in range(tidy + kernel_r, h - kernel_r, ntidy):
        for i in range(tidx + kernel_r, w - kernel_r, ntidx):
            if mask[j, i] == 0.0:
                count = 0.0
                avg_c0 = 0.0
                avg_c1 = 0.0
                avg_c2 = 0.0
                for y in range(-kernel_r, kernel_r + 1):
                    for x in range(-kernel_r, kernel_r + 1):
                        if kernel[y + kernel_r, x + kernel_r] > 0.0 and mask[j + y, i + x] > 0.0:
                            count += 1.0
                            avg_c0 = avg_c0 * (count-1) / count + src[j + y, i + x, 0] / count
                            avg_c1 = avg_c1 * (count-1) / count + src[j + y, i + x, 1] / count
                            avg_c2 = avg_c2 * (count-1) / count + src[j + y, i + x, 2] / count
                oj = j - kernel_r
                oi = i - kernel_r
                out[oj, oi, 0] = avg_c0
                out[oj, oi, 1] = avg_c1
                out[oj, oi, 2] = avg_c2


def blur_cpu(src, mask, kernel, kernel_s, w, h):
    import multiprocessing
    from joblib import Parallel, delayed
    # this is safe, right? ( --___-- );
    # high kernel size can take forever to compute & idk how to abort it midway yet;
    # prob. a good idea to avoid using all cores.

    # region auxiliary methods

    def calculate_average_color(src, mask, kernel_indices, j, i):
        valid_indices = kernel_indices + [j, i]
        valid_pixels = mask[valid_indices[:, 0], valid_indices[:, 1]] > 0.0
        if valid_pixels.any():
            avg_c = np.mean(src[valid_indices[valid_pixels, 0], valid_indices[valid_pixels, 1]], axis=0)
        else:
            avg_c = np.zeros(3)
        return avg_c

    def process_chunk(args):
        src, mask, kernel_indices, kernel_s, w, h, start_row, end_row = args
        kernel_r = kernel_s // 2
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
    kernel_indices = np.transpose(np.nonzero(kernel)) - kernel_s // 2
    num_cores = multiprocessing.cpu_count()
    num_cores = max(1, num_cores-1)  # leave 1 core unoccupied in case computations take forever
    chunk_size = h // num_cores

    chunks = [(src, mask, kernel_indices, kernel_s, w, h, i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    # include any leftover rows
    if h % num_cores != 0:
        chunks[-1] = (src, mask, kernel_indices, kernel_s, w, h, (num_cores - 1) * chunk_size, h)

    # compute using available cores
    with Parallel(n_jobs=num_cores) as parallel:
        out_chunks = parallel(delayed(process_chunk)(chunk) for chunk in chunks)

    out = np.vstack(out_chunks)   # combine results

    return out
