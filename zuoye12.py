import cv2
import numpy as np
import gradio as gr

points_src = []
points_dst = []
image = None


def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img


def record_points(evt: gr.SelectData):
    global points_src, points_dst, image

    if image is None:
        return None

    x, y = evt.index[0], evt.index[1]

    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 3, (255, 0, 0), -1)  # Blue for source (放大为3提升可见度)
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 3, (0, 0, 255), -1)  # Red for target

    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 2)

    return marked_image


def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image using MLS Affine Deformation.
    """
    if image is None:
        return None

    warped_image = np.array(image)

    n_points = min(len(source_pts), len(target_pts))
    if n_points < 3:
        print("Please provide at least 3 pairs of points for deformation.")
        return warped_image

    source_pts = source_pts[:n_points]
    target_pts = target_pts[:n_points]


    h, w = warped_image.shape[:2]

    p = target_pts.astype(np.float64)
    q = source_pts.astype(np.float64)

    grid_X, grid_Y = np.meshgrid(np.arange(w), np.arange(h))
    v = np.stack([grid_X, grid_Y], axis=-1).astype(np.float64)

    p_reshaped = p[np.newaxis, np.newaxis, :, :]
    v_reshaped = v[:, :, np.newaxis, :]

    dist_sq = np.sum((v_reshaped - p_reshaped) ** 2, axis=-1)
    weights = 1.0 / (dist_sq ** alpha + eps)
    sum_weights = np.sum(weights, axis=-1, keepdims=True)

    p_star = np.sum(weights[..., np.newaxis] * p_reshaped, axis=2) / sum_weights
    q_reshaped = q[np.newaxis, np.newaxis, :, :]
    q_star = np.sum(weights[..., np.newaxis] * q_reshaped, axis=2) / sum_weights

    phat = p_reshaped - p_star[:, :, np.newaxis, :]
    qhat = q_reshaped - q_star[:, :, np.newaxis, :]


    M = np.einsum('hwn, hwni, hwnj -> hwij', weights, phat, phat)


    M[..., 0, 0] += eps
    M[..., 1, 1] += eps


    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    det = a * d - b * c

    inv_M = np.empty_like(M)
    inv_M[..., 0, 0] = d / det
    inv_M[..., 0, 1] = -b / det
    inv_M[..., 1, 0] = -c / det
    inv_M[..., 1, 1] = a / det


    A = np.einsum('hwn, hwni, hwnj -> hwij', weights, phat, qhat)


    M_inv_A = np.einsum('hwij, hwjk -> hwik', inv_M, A)


    v_minus_pstar = v - p_star
    offset = np.einsum('hwi, hwij -> hwj', v_minus_pstar, M_inv_A)
    map_xy = q_star + offset


    map_x = map_xy[..., 0].astype(np.float32)
    map_y = map_xy[..., 1].astype(np.float32)


    warped_image = cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return warped_image


def run_warping():
    global points_src, points_dst, image

    if image is None:
        return None


    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image



def clear_points():
    global points_src, points_dst, image
    points_src.clear()
    points_dst.clear()
    return image



with gr.Blocks() as demo:
    gr.Markdown("## Point-Guided Image Warping Playground (MLS)")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points (Blue=Src, Red=Dst)",
                                    interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    with gr.Row():
        run_button = gr.Button("Run Warping", variant="primary")
        clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

if __name__ == "__main__":
    demo.launch()