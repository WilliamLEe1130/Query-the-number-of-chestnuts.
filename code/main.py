import cv2
import utils
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__":
    #基础配置
    img_path = "D:/Pycharmprojects/Query-the-number-of-chestnuts/imags/chesnuts.png"
    save_dir = "D:/Pycharmprojects/Query-the-number-of-chestnuts/imags"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #全流程处理
    # 步骤1：原始图像
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise ValueError(f"无法读取图像，请检查路径：{img_path}")
    cv2.imwrite(os.path.join(save_dir, "1_raw_img.jpg"), raw_img)
    print("✅ 步骤1：原始图像已保存")

    # 步骤2：HSV颜色筛选
    img_blur = cv2.GaussianBlur(raw_img, (7, 7), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([5, 50, 40])
    upper_brown = np.array([35, 230, 200])
    hsv_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    cv2.imwrite(os.path.join(save_dir, "2_hsv_mask_img.jpg"), hsv_mask)
    print("✅ 步骤2：HSV颜色筛选掩码已保存")

    # 步骤3：预处理最终二值图（孔洞填充+形态学优化）
    processed_img, preprocessed_mask = utils.preprocess_image(img_path)
    cv2.imwrite(os.path.join(save_dir, "3_preprocessed_mask_img.jpg"), preprocessed_mask)
    print("✅ 步骤3：预处理最终二值图已保存（黑白分明）")

    # 步骤4：形态学操作（分离粘连）
    morph_mask = utils.morphological_operation(preprocessed_mask)
    cv2.imwrite(os.path.join(save_dir, "4_morph_mask_img.jpg"), morph_mask)
    print("✅ 步骤4：形态学操作后图像已保存")

    # 步骤5：距离变换图
    dist_transform = cv2.distanceTransform(morph_mask, cv2.DIST_L2, 5)
    dist_transform_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(save_dir, "5_dist_transform_norm_img.jpg"), dist_transform_norm)
    print("✅ 步骤5：距离变换图已保存")

    # 步骤6：分水岭分割
    segmentation_img = raw_img.copy()
    segmentation_img, markers = utils.watershed_segmentation(segmentation_img, morph_mask)
    cv2.imwrite(os.path.join(save_dir, "6_segmentation_img.jpg"), segmentation_img)
    print("✅ 步骤6：分水岭分割结果已保存")

    # 调试信息
    has_boundary = -1 in markers
    print(f"\n🔍 分水岭是否生成边界标记：{has_boundary}")
    if has_boundary:
        print(f"🔍 边界像素数量：{np.sum(markers == -1)}")
    print(f"🔍 预处理后二值图黑白占比：白像素{np.sum(preprocessed_mask == 255)}个，黑像素{np.sum(preprocessed_mask == 0)}个")

    # ===================== 3. 统计结果 =====================
    chestnut_count, pixel_counts = utils.count_and_calculate_pixels(markers)
    print("\n" + "="*50)
    print(f"📊 糖炒栗子总数：{chestnut_count}")
    print(f"📊 每个栗子的像素数：{pixel_counts}")
    print(f"📊 平均每个栗子像素数：{sum(pixel_counts)/len(pixel_counts):.0f}")
    print("="*50 + "\n")

    # ===================== 4. 可视化（新增预处理中间步骤）=====================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"糖炒栗子图像处理全流程（总数：{chestnut_count}）", fontsize=16)

    # 1. 原始图像
    axes[0, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. 原始图像")
    axes[0, 0].axis("off")

    # 2. HSV颜色筛选掩码
    axes[0, 1].imshow(hsv_mask, cmap="gray")
    axes[0, 1].set_title("2. HSV颜色筛选掩码")
    axes[0, 1].axis("off")

    # 3. 预处理最终二值图
    axes[0, 2].imshow(preprocessed_mask, cmap="gray")
    axes[0, 2].set_title("3. 预处理最终二值图（黑白分明）")
    axes[0, 2].axis("off")

    # 4. 形态学操作后
    axes[1, 0].imshow(morph_mask, cmap="gray")
    axes[1, 0].set_title("4. 形态学操作后")
    axes[1, 0].axis("off")

    # 5. 距离变换图
    axes[1, 1].imshow(dist_transform_norm, cmap="gray")
    axes[1, 1].set_title("5. 距离变换图")
    axes[1, 1].axis("off")

    # 6. 分割结果
    axes[1, 2].imshow(cv2.cvtColor(segmentation_img, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"6. 分水岭分割结果（红边+白点）")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "7_Full-process_visualization_summary.jpg"), dpi=300, bbox_inches="tight")
    plt.show()

