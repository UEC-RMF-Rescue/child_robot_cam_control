import cv2
import numpy as np

# 画像の読み込み
image_path = r"C:\Users\yff76\Lecture Document\omiya.jpg"  # raw文字列を使用
image = cv2.imread(image_path)

# 画像が正しく読み込まれたか確認
if image is None:
    print(f"Error: Could not open or find the image '{image_path}'.")
    exit()

# 入力画像上の4点を指定
# これらの点は、画像上の実際の座標に対応する必要があります
pts_src = np.array([[0, 100], [2200, 0],[0, 2300],  [2400, 2500]], dtype='float32')

# 出力画像上の4点を指定
# ここでは、画像のサイズを(500, 500)に指定しています
pts_dst = np.array([[0, 0], [2500, 0], [0, 2500], [2500, 2500]], dtype='float32')

# 透視変換行列を計算
M = cv2.getPerspectiveTransform(pts_src, pts_dst)

# 画像を透視変換
transformed = cv2.warpPerspective(image, M, (0, 0))

# 画像を縮小
scale_percent = 30  # 縮小する割合（パーセント）
width = int(transformed.shape[1] * scale_percent / 100)
height = int(transformed.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(transformed, dim, interpolation=cv2.INTER_AREA)

# 結果の画像を保存
cv2.imwrite('output.jpg', resized)

# 結果の画像を表示
cv2.imshow('Transformed and Resized Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
