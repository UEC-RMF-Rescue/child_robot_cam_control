import cv2
import numpy as np

def detect_lines(image_path, scale_percent=50, output_scale_percent=50, top_n=10, overlap_threshold=0.5):
    # 画像を読み込む
    image = cv2.imread(image_path)
    
    # 画像のサイズを縮小
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Canny エッジ検出
    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)
    
    # Hough 変換で直線検出
    min_line_length = 100
    max_line_gap = 10
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is not None:
        # 各直線を表示
        lines = np.array([line[0] for line in lines])
        line_density = np.zeros_like(gray, dtype=np.float32)
        
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_density, (x1, y1), (x2, y2), 1, 1)
        
        # 線密度を計算し、閾値を超えるエリアを検出
        density_map = np.clip(line_density, 0, 1)
        _, binary_density = cv2.threshold(density_map, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_density.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                rectangles.append((x, y, x+w, y+h))
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
            
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if centers:
            for idx, center in enumerate(centers):
                cx, cy = center
                cv2.circle(resized_image, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(resized_image, f"{idx+1}", (cx - 15, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(resized_image, f"({cx}, {cy})", (cx + 20, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                print(f"Detected combined rectangle with center at ({cx}, {cy})")
    
    density_color_map = cv2.applyColorMap((line_density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    output_width = int(resized_image.shape[1] * output_scale_percent / 100)
    output_height = int(resized_image.shape[0] * output_scale_percent / 100)
    output_dim = (output_width, output_height)
    
    output_image = cv2.resize(resized_image, output_dim, interpolation=cv2.INTER_AREA)
    density_map_resized = cv2.resize(density_color_map, output_dim, interpolation=cv2.INTER_AREA)
    
    cv2.imshow('Detected Rectangles and Coordinates', output_image)
    cv2.imshow('Line Density Map', density_map_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画像ファイルのパスを指定
image_path = r"C:\Users\yff76\Lecture Document\DSC_1146.JPG"
detect_lines(image_path, scale_percent=50, output_scale_percent=50, top_n=10, overlap_threshold=0.5)
