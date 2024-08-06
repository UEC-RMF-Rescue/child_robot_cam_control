import cv2
import numpy as np

# YOLOの設定ファイルと重みファイルのパス
yolo_config_path = r"C:\Users\yff76\Lecture Document\yolov3.cfg"
yolo_weights_path = r"C:\Users\yff76\Lecture Document\yolov3.weights"
yolo_names_path = r"C:\Users\yff76\Lecture Document\coco.names"

# ラベルの読み込み
with open(yolo_names_path, 'r') as f:
    labels = f.read().strip().split('\n')

# YOLOモデルの読み込み
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

# 画像の読み込み
image_path = r"C:\Users\yff76\Lecture Document\DSC_1100.JPG" 
image = cv2.imread(image_path)
(H, W) = image.shape[:2]

# 画像の縮小表示のためのリサイズ
width = 1000
ratio = width / image.shape[1]
height = int(image.shape[0] * ratio)
resized_image = cv2.resize(image, (width, height))

(H, W) = resized_image.shape[:2]

# YOLO用に画像を前処理
blob = cv2.dnn.blobFromImage(resized_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# YOLOネットワークの出力レイヤー名を取得
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 物体検出の実行
layer_outputs = net.forward(output_layers)

# 検出結果の初期化
boxes = []
confidences = []
class_ids = []

# 各検出結果を解析
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0:  # 信頼度のしきい値を設定
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 重複するボックスを削除
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 検出結果を画像に描画
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in np.random.randint(0, 255, size=(3,), dtype="uint8")]
        cv2.rectangle(resized_image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
        cv2.putText(resized_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 結果を表示
cv2.imshow("Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()