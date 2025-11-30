import cv2
import numpy as np

def preprocess_image(img_path):
    """优化后预处理：精准颜色筛选+二值化增强+孔洞填充，确保黑白分明"""
    # 1. 读取图像（保留原始彩色图用于后续分割绘制）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径：{img_path}")

    # 2. 预处理：去噪（先模糊再处理，减少反光干扰）
    img_blur = cv2.GaussianBlur(img, (3, 5), 0)  # 7x7高斯核，更强去噪

    # 3. 颜色空间转换：RGB→HSV（更适合颜色筛选）
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # 4. 优化HSV阈值（适配深褐色栗子+浅色托盘，扩大颜色范围，排除反光）
    # 深褐色HSV范围（色相0-40，饱和度50-250，明度44-200）自己调节选出的最优阈值范围
    lower_brown = np.array([0, 50, 44])
    upper_brown = np.array([40, 250, 200])

    # 5. 颜色筛选：得到初步掩码（栗子区域为白，背景为黑）
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # 6. 二值化增强：强制黑白分明（避免灰色过渡区域）
    # 大津法自动阈值 + 反色（确保栗子是白，背景是黑）
    ret, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 7. 孔洞填充：修复栗子内部因反光导致的黑色孔洞（关键！）
    # 步骤：先找到所有轮廓，再填充轮廓内部
    contours, _ = cv2.findContours(mask_bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask_bin, [contour], -1, 255, -1)  # -1表示填充内部

    # 8. 形态学优化：去除微小噪点（背景中的小黑点）
    kernel_small = np.ones((5, 5), np.uint8)
    # 开运算（先腐蚀再膨胀）：去除小噪点，保留大目标
    mask_morph = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # 9. 关键：面积筛选（保留扁平栗子，过滤极小噪点）
    mask_final = np.zeros_like(mask_morph)
    min_area = 100  # 适配扁平栗子的小面积（可根据实际调整）
    contours_final, _ = cv2.findContours(mask_morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_final:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(mask_final, [contour], -1, 255, -1)
    return img, mask_final  # 返回原始彩色图和最终优化二值图


def morphological_operation(mask):
    """形态学操作：分离粘连，优化轮廓"""
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((7, 7), np.uint8)
    # 先腐蚀（去噪+细化粘连），再膨胀（修复轮廓）
    erode = cv2.erode(mask, kernel1, iterations=5)
    dilate = cv2.dilate(erode, kernel2, iterations=1)
    # 闭运算（先膨胀再腐蚀）：进一步填充细小缝隙
    close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel2, iterations=1)
    return close


def watershed_segmentation(img, mask):
    """分水岭分割：优化标记生成，确保红色边界显示"""
    kernel = np.ones((5, 5), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.42 * dist_transform.max(), 255, 0)
    #经调节，0.42为最优值
    sure_fg = np.uint8(sure_fg)

    ret, sure_bg = cv2.threshold(sure_bg, 127, 255, 0)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    cv2.watershed(img, markers)
    # 强制绘制红色边界（厚度2，更清晰）
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)  # 厚度改为3，红色更醒目
    # 中心白点（方便核对）
    for i in range(2, markers.max() + 1):
        center = np.where(markers == i)
        if len(center[0]) > 0 and len(center[1]) > 0:
            cx = int(np.mean(center[1]))
            cy = int(np.mean(center[0]))
            cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)

    return img, markers


def count_and_calculate_pixels(markers):
    """统计栗子数量和每个的像素数"""
    marker_ids = np.unique(markers)
    # 过滤背景（0）、边界（-1）、无效标记（1，可能是背景噪声）
    valid_ids = [id for id in marker_ids if id not in (-1, 0, 1)]
    count = len(valid_ids)
    pixel_counts = [np.sum(markers == id) for id in valid_ids]
    return count, pixel_counts