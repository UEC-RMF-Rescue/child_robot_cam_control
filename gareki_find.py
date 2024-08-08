import numpy as np
import cv2

#このプログラムは指定した画像から直線成分を抽出して瓦礫がありそうな場所を最大3つまで出力します。
#画像上に瓦礫の中心位置と推測できる点と、そのエリアを長方形で描画します。
#また、線密度の分布も表示します。
#最後に画像上の瓦礫の位置の座標と実座標に変換した座標を表示します。

def detect_lines_and_transform(image_path, scale_percent=50, output_scale_percent=50, top_n=3):
    # 画像を読み込む
    image = cv2.imread(image_path)
    
    # 画像のサイズを縮小
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Canny エッジ検出
    low_threshold = 50   # 閾値を緩くする
    high_threshold = 150 # 閾値を緩くする
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)
    
    # Hough 変換で直線検出
    min_line_length = 50  # 最小線長を減らす
    max_line_gap = 20     # 最大線ギャップを増やす
    threshold = 50        # 最小票数を減らす
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    line_density = np.zeros_like(gray, dtype=np.float32)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_density, (x1, y1), (x2, y2), 1, 1)
        
        # 線密度を計算し、閾値を超えるエリアを検出
        density_map = np.clip(line_density, 0, 1)
        _, binary_density = cv2.threshold(density_map, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_density.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                rectangles.append((x, y, x+w, y+h))
        
        def is_overlapping(rect1, rect2):
            x1, y1, x2, y2 = rect1
            x3, y3, x4, y4 = rect2
            return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)
        
        #隣接長方形をくっつけて大きな瓦礫とする
        def merge_rectangles(rect_list):
            merged = []
            while rect_list:
                rect = rect_list.pop(0)
                merge_group = [rect]
                for other in rect_list[:]:
                    if is_overlapping(rect, other):
                        merge_group.append(other)
                        rect_list.remove(other)
                x1 = min([r[0] for r in merge_group])
                y1 = min([r[1] for r in merge_group])
                x2 = max([r[2] for r in merge_group])
                y2 = max([r[3] for r in merge_group])
                merged.append((x1, y1, x2, y2))
            return merged
        
        merged_rectangles = merge_rectangles(rectangles)
        
        # 高密度エリアをソート
        merged_rectangles.sort(key=lambda r: (r[2]-r[0]) * (r[3]-r[1]), reverse=True)
        
        centers = []
        for idx, rect in enumerate(merged_rectangles[:top_n]):
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centers.append((cx, cy))
            
            # 上位3エリアに異なる色を付ける
            color = (0, 255, 0) if idx == 0 else (255, 0, 0) if idx == 1 else (0, 0, 255)
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), color, 2)
        
        if centers:
            for idx, center in enumerate(centers):
                cx, cy = center
                cv2.circle(resized_image, (cx, cy), 7, (0, 0, 255), -1)
                cv2.putText(resized_image, f"{idx+1}", (cx - 15, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(resized_image, f"({cx}, {cy})", (cx + 20, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 画像座標から実座標に変換
                pts_image = np.array([[cx, cy]], dtype=np.float32)
                pts_image = np.array([pts_image])
                pts_real = cv2.perspectiveTransform(pts_image, M)
                
                real_x, real_y = pts_real[0][0]
                print(f"Detected combined rectangle with center at image coordinates ({cx}, {cy})")
                print(f"Real world coordinates: ({real_x:.2f}, {real_y:.2f})")
    
    else:
        print("No lines detected")
    
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

# 射影変換のための座標
pts1 = np.array([(171, 275), (434, 272), (63, 397), (488, 405)], dtype=np.float32)
pts2 = np.array([(-434, 1520), (175, 1520), (-434, 912), (175, 912)], dtype=np.float32)

# 射影行列の取得
M = cv2.getPerspectiveTransform(pts1, pts2)

# 画像ファイルのパスを指定
image_path = r"C:\Users\yff76\Lecture Document\photo_11.jpg"
detect_lines_and_transform(image_path, scale_percent=100, output_scale_percent=50)
