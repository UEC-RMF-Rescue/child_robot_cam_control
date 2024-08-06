import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def detect_cuboids(image_path, scale_percent=50, min_area=2000, proximity_threshold=50, output_scale_percent=50, overlap_threshold=0.5):
    # 画像を読み込む
    image = cv2.imread(image_path)
    
    # 画像のサイズを縮小
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        lines = np.array([line[0] for line in lines])
    
        # 直線を画像に描画
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 直線が集中している領域を見つけるための画像を作成
        line_density = np.zeros_like(gray, dtype=np.float32)
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_density, (x1, y1), (x2, y2), 1, 1)
        
        # 直線の密度を計算し、閾値を超えるエリアを検出
        _, binary_density = cv2.threshold(line_density, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_density.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:  # 大きさのフィルタリング
                x, y, w, h = cv2.boundingRect(cnt)
                rectangles.append((x, y, x+w, y+h))
                # 長方形の描画
                cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        def calculate_overlap(rect1, rect2):
            x1, y1, x2, y2 = rect1
            x3, y3, x4, y4 = rect2
            overlap_x1 = max(x1, x3)
            overlap_y1 = max(y1, y3)
            overlap_x2 = min(x2, x4)
            overlap_y2 = min(y2, y4)
            overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)
            return overlap_area / min(area1, area2)
        
        def merge_rectangles(rect_list):
            merged = []
            while rect_list:
                rect = rect_list.pop(0)
                merge_group = [rect]
                for other in rect_list[:]:
                    if calculate_overlap(rect, other) > overlap_threshold:
                        merge_group.append(other)
                        rect_list.remove(other)
                x1 = min([r[0] for r in merge_group])
                y1 = min([r[1] for r in merge_group])
                x2 = max([r[2] for r in merge_group])
                y2 = max([r[3] for r in merge_group])
                merged.append((x1, y1, x2, y2))
            return merged
        
        merged_rectangles = merge_rectangles(rectangles)
        
        centers = []
        for rect in merged_rectangles:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((cx, cy))
            
            # 統合された長方形の描画
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if centers:
            # DBSCANで近くの中心をまとめる
            centers_np = np.array(centers)
            clustering = DBSCAN(eps=proximity_threshold, min_samples=1).fit(centers_np)
            labels = clustering.labels_
            
            unique_labels = np.unique(labels)
            for idx, label in enumerate(unique_labels):
                cluster_points = centers_np[labels == label]
                cluster_center = np.mean(cluster_points, axis=0).astype(int)
                
                # 点に番号を振る
                cv2.circle(resized_image, tuple(cluster_center), 7, (0, 0, 255), -1)
                cv2.putText(resized_image, f"{idx+1}", (cluster_center[0] - 15, cluster_center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(resized_image, f"({cluster_center[0]}, {cluster_center[1]})", (cluster_center[0] + 20, cluster_center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 座標をコンソールに出力
                print(f"Detected combined cuboid with center at ({cluster_center[0]}, {cluster_center[1]})")

    # 出力画像の縮小
    output_width = int(resized_image.shape[1] * output_scale_percent / 100)
    output_height = int(resized_image.shape[0] * output_scale_percent / 100)
    output_dim = (output_width, output_height)
    
    output_image = cv2.resize(resized_image, output_dim, interpolation=cv2.INTER_AREA)
    
    # 画像を表示
    cv2.imshow('Detected Cuboids', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像ファイルのパスを指定
image_path = r"C:\Users\yff76\Lecture Document\DSC_1146.JPG"
detect_cuboids(image_path, scale_percent=50, min_area=2000, proximity_threshold=50, output_scale_percent=50, overlap_threshold=0.5)
