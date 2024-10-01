import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import csv

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None



# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
  
    # warped_image = np.array(image)
    # ### FILL: 基于MLS or RBF 实现 image warping
    h, w, _ = image.shape
    warped_image = np.zeros_like(image)
    # 交换 source_pts 和 target_pts, 实现向后映射
    source_pts, target_pts = target_pts.copy(), source_pts.copy()
    VVV = []
    # 遍历图像中的每个像素
    for y in range(h):
        for x in range(w):
            v = np.array([x, y])
            
            # 计算 p* 和 q*（控制点和目标点的加权中心）
            weights = np.array([1 / ((np.linalg.norm(p - v) + eps) ** 2) for p in source_pts])
            p_star = np.sum(weights[:, np.newaxis] * source_pts, axis=0) / np.sum(weights)
            q_star = np.sum(weights[:, np.newaxis] * target_pts, axis=0) / np.sum(weights)
            
            # 计算 \hat{p_i} 和 \hat{q_i}（点到中心的偏移）
            p_hat = source_pts - p_star
            q_hat = target_pts - q_star
            
            # 计算中间矩阵并求逆
            p_hat_w = np.array([w * np.outer(ph, ph) for w, ph in zip(weights, p_hat)])
            A = np.sum(p_hat_w, axis=0)
            if np.abs(np.linalg.det(A)) < eps:
                v_prime = v
            else:
                A_inv = np.linalg.inv(A)
                q_hat_w = np.array([w * np.outer(ph, qh) for w, ph, qh in zip(weights, p_hat, q_hat)])
                B = np.sum(q_hat_w, axis=0)
                # 计算新的点
                # v_prime = (v - p_star).dot(A_inv).dot(np.sum([w * np.outer(ph, qh) for w, ph, qh in zip(weights, p_hat, q_hat)], axis=0)) + q_star
                v_prime = (v - p_star).dot(A_inv).dot(B) + q_star
            VVV.append(v_prime)

            # if 0 <= v_prime[0] < w and 0 <= v_prime[1] < h:
            #     warped_image[int(v_prime[1]), int(v_prime[0])] = image[y, x]
            if 0 <= v_prime[0] < w - 1 and 0 <= v_prime[1] < h - 1:
                x0 = int(np.floor(v_prime[0]))
                y0 = int(np.floor(v_prime[1]))
                x1 = x0 + 1
                y1 = y0 + 1

                dx = v_prime[0] - x0
                dy = v_prime[1] - y0

                # 获取四个邻近像素值
                I00 = image[y0, x0]  # 左上
                I10 = image[y0, x1]  # 右上
                I01 = image[y1, x0]  # 左下
                I11 = image[y1, x1]  # 右下

                # 进行双线性插值
                warped_pixel = (
                    (1 - dx) * (1 - dy) * I00
                    + dx * (1 - dy) * I10
                    + (1 - dx) * dy * I01
                    + dx * dy * I11
                )

                warped_image[y, x] = warped_pixel
            else:
                # 映射坐标超出范围, 填充黑色
                warped_image[y, x] = [255, 255, 255]


   
    np.savetxt(
    "transformed_points.csv", np.array(VVV).reshape(-1, 2), delimiter=","
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
