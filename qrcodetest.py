import cv2
import numpy as np

# -----------------------------------------------------------
# initial
# -----------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
FILE_PNG_AB = r"C:\Users\yff76\Lecture Document\testqr5.JPG"
SCALE_PERCENT = 50  # 画像サイズを50%に縮小
ZOOM_FACTOR = 2  # QRコードが検出された場所の拡大倍率

# -----------------------------------------------------------
# function to compress image
# -----------------------------------------------------------
def compress_image(img, scale_percent):
    # 現在の画像サイズを取得
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # 画像サイズを変更
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

# -----------------------------------------------------------
# function to preprocess image
# -----------------------------------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # グレースケール変換
    equalized = cv2.equalizeHist(gray)  # ヒストグラム均一化
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)  # ガウシアンブラー
    edged = cv2.Canny(blurred, 50, 150)  # エッジ検出
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)  # モルフォロジー変換
    return morphed

# -----------------------------------------------------------
# function to detect QR codes
# -----------------------------------------------------------
def function_qrdec_cv2(img_bgr):
    # QRCodeDetectorインスタンス生成
    qrd = cv2.QRCodeDetector()

    # QRコードデコード
    retval, decoded_info, points, straight_qrcode = qrd.detectAndDecodeMulti(img_bgr)

    if retval:
        points = points.astype(np.int32)  # Change np.int to np.int32

        for dec_inf, point in zip(decoded_info, points):
            if dec_inf == '':
                continue

            # QRコード座標取得
            x = point[0][0]
            y = point[0][1]

            # QRコードデータ (SHIFT-JISとしてデコード)
            try:
                dec_inf_shiftjis = dec_inf.encode('latin1').decode('shift_jis')
            except UnicodeDecodeError:
                dec_inf_shiftjis = dec_inf

            print('dec:', dec_inf_shiftjis)
            # QRコードの座標を出力
            print('QRコードの座標:')
            for coord in point:
                print(f'({coord[0]}, {coord[1]})')

            # OpenCVのputText関数を使って日本語を画像に描画する
            img_bgr = cv2.putText(img_bgr, dec_inf_shiftjis, (x, y-6), font, .6, (0, 0, 255), 2, cv2.LINE_AA)

            # バウンディングボックス
            img_bgr = cv2.polylines(img_bgr, [point], True, (0, 255, 0), 2, cv2.LINE_AA)

            # QRコードの中心に円を描く
            center_x = int(np.mean(point[:, 0]))
            center_y = int(np.mean(point[:, 1]))
            img_bgr = cv2.circle(img_bgr, (center_x, center_y), 5, (255, 0, 0), -1)

            # QRコードがある部分を拡大して表示
            min_x = np.min(point[:, 0])
            max_x = np.max(point[:, 0])
            min_y = np.min(point[:, 1])
            max_y = np.max(point[:, 1])

            qr_region = img_bgr[min_y:max_y, min_x:max_x]
            qr_region_zoomed = cv2.resize(qr_region, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LINEAR)

            cv2.imshow('Zoomed QR Code', qr_region_zoomed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    cv2.imshow('image', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------------------------------------
# sample program
# -----------------------------------------------------------
img_BGR = cv2.imread(FILE_PNG_AB, cv2.IMREAD_COLOR)
if img_BGR is not None:
    resized_img_BGR = compress_image(img_BGR, SCALE_PERCENT)
    function_qrdec_cv2(resized_img_BGR)
else:
    print(f"Error: Unable to read image {FILE_PNG_AB}")
