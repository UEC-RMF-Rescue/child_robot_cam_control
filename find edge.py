import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image_path = r"C:\Users\yff76\Lecture Document\omiya.jpg"  # raw文字列を使用
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Cannyエッジ検出
edges = cv2.Canny(image, 100, 200)

# 元の画像とエッジ検出結果をウィンドウに表示
cv2.imshow('Original Image', image)
cv2.imshow('Edge Image', edges)

# キー入力を待つ（任意のキーが押されるまで）
cv2.waitKey(0)

# ウィンドウを閉じる
cv2.destroyAllWindows()
