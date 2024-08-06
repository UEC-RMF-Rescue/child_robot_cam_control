import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread(r"C:\Users\yff76\Lecture Document\DSC_1096.JPG")

# 画像の縮小
scale_percent = 30  # 画像を50%のサイズに縮小
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# 画像をぼかす
##blurred = cv2.GaussianBlur(resized, (5, 5), 0)

# グレースケール変換
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# 前処理 - ヒストグラム均等化
equalized = cv2.equalizeHist(gray)

# エッジ検出
edges = cv2.Canny(equalized, 50, 150)

# 直線検出 (Hough変換)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

def find_rectangles(lines):
    rectangles = []
    if lines is None:
        return rectangles

    for line in lines:
        x1, y1, x2, y2 = line[0]
        rectangles.append(((x1, y1), (x2, y2)))

    return rectangles

def filter_rectangles(rectangles, min_area, min_length, aspect_ratio_range):
    filtered = []
    for rect in rectangles:
        p1, p2 = rect
        width = abs(p2[0] - p1[0])
        height = abs(p2[1] - p1[1])
        area = width * height
        aspect_ratio = width / height if height != 0 else 0
        
        if area >= min_area and width >= min_length and height >= min_length:
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                filtered.append(rect)
    
    return filtered

# 条件設定
min_area = 1000           # 最小面積
min_length = 50           # 最小辺の長さ
aspect_ratio_range = (0.5, 2.0)  # 縦横比の範囲

# 長方形の検出とフィルタリング
rectangles = find_rectangles(lines)
filtered_rectangles = filter_rectangles(rectangles, min_area, min_length, aspect_ratio_range)

# 長方形の座標を保存する配列
rectangles_coordinates = []

for rect in filtered_rectangles:
    p1, p2 = rect
    cv2.rectangle(resized, p1, p2, (255, 0, 0), 2)
    rectangles_coordinates.append((p1, p2))

def calculate_average_coordinates(rectangles_coords):
    if not rectangles_coords:
        return None

    x_coords = []
    y_coords = []
    
    for rect in rectangles_coords:
        p1, p2 = rect
        x_coords.extend([p1[0], p2[0]])
        y_coords.extend([p1[1], p2[1]])
    
    avg_x = np.mean(x_coords)
    avg_y = np.mean(y_coords)
    
    return (avg_x, avg_y)

def group_rectangles_by_distance(rectangles_coords, distance_threshold):
    grouped_rectangles = []
    visited = [False] * len(rectangles_coords)

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def dfs(index, group):
        stack = [index]
        while stack:
            i = stack.pop()
            if not visited[i]:
                visited[i] = True
                group.append(rectangles_coords[i])
                for j in range(len(rectangles_coords)):
                    if not visited[j] and distance(rectangles_coords[i][0], rectangles_coords[j][0]) < distance_threshold:
                        stack.append(j)

    for i in range(len(rectangles_coords)):
        if not visited[i]:
            group = []
            dfs(i, group)
            grouped_rectangles.append(group)

    return grouped_rectangles

# 距離閾値の設定
distance_threshold = 100  # 長方形間の最大距離

# 長方形を距離に基づいてグループ化
grouped_rectangles = group_rectangles_by_distance(rectangles_coordinates, distance_threshold)

# 各グループの平均座標を計算
average_coordinates_per_group = [calculate_average_coordinates(group) for group in grouped_rectangles]

# 結果を表示
cv2.imshow('Filtered Rectangles', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 各グループの平均座標を出力
for i, avg_coords in enumerate(average_coordinates_per_group):
    if avg_coords:
        print(f"Group {i+1} Average Coordinates: {avg_coords}")
    else:
        print(f"Group {i+1}: No rectangles detected.")
