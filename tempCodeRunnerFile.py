import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter

def detect_cuboids(image_path, scale_percent=100, min_area=2000, proximity_threshold=50, output_scale_percent=50, density_threshold=0.5, smoothing_sigma=1.0, min_line_count=10):
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
    
    if lines is not None and len(lines) > 0:
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
        
        # ガウシアンフィルタで線密度をスムージング
        smoothed_density = gaussian_filter(line_density, sigma=smoothing_sigma)
        
        # 密度が高いエリアを検出
        _, binary_density = cv2.threshold(smoothed_density, density_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_density.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:  # 大きさのフィルタリング
                x, y, w, h = cv2.boundingRect(cnt)
                rectangles.append((x, y, x+w, y+h))
                areas.append(area)
                # 長方形の描画
                cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # 線がほとんどない場合、直方体を検出
        if lines is None or len(lines) == 0:
            contours, _ = cv2.findContours(binary_density.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    rectangles.append((x, y, x+w, y+h))
                    areas.append(area)
                    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            # 各領域内の線の本数をカウントし、その本数に基づいてフィルタリング
            filtered_rectangles = []
            filtered_areas = []
            for rect in rectangles:
                x1, y1, x2, y2 = rect
                rect_area = line_density[y1:y2, x1:x2]
                line_count = np.sum(rect_area > 0)
                
                if line_count >= min_line_count:
                    filtered_rectangles.append(rect)
                    filtered_areas.append(np.sum(rect_area > 0))
                    
            # 密度の高い領域を上位10個に絞る
            sorted_indices = np.argsort(filtered_areas)[::-1]
            top_indices = sorted_indices[:10]
            
            top_rectangles = [filtered_rectangles[i] for i in top_indices]
            top_areas = [filtered_areas[i] for i in top_indices]
            
            centers = []
            for rect in top_rectangles:
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
image_path = r"C:\Users\yff76\Lecture Document\DSC_1141.JPG"
detect_cuboids(image_path, scale_percent=50, min_area=2000, proximity_threshold=50, output_scale_percent=50, density_threshold=0.5, smoothing_sigma=1.0, min_line_count=10)
