import cv2
import numpy as np

# -----------------------------------------------------------
# initial
# -----------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
FILE_PNG_AB = r"C:\Users\yff76\Lecture Document\DSC_1151.JPG"
SCALE_PERCENT = 50  # 画像サイズを50%に縮小
ZOOM_FACTOR = 2  # QRコードが検出された場所の拡大倍率
SHOW_SCALE_PERCENT = 25  # 最終画像を縮小表示する倍率
ROTATION_ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,]

# -----------------------------------------------------------
# function to compress image
# -----------------------------------------------------------
def compress_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img

# -----------------------------------------------------------
# function to preprocess image
# -----------------------------------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    return morphed

# -----------------------------------------------------------
# function to rotate image
# -----------------------------------------------------------
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# -----------------------------------------------------------
# function to detect QR codes
# -----------------------------------------------------------
def function_qrdec_cv2(img_bgr):
    qrd = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qrd.detectAndDecodeMulti(img_bgr)
    qr_codes = []
    if retval:
        points = np.array(points)
        for dec_inf, point in zip(decoded_info, points):
            if dec_inf == '':
                continue
            point = point.astype(np.int32)
            center_x = int(np.mean(point[:, 0]))
            center_y = int(np.mean(point[:, 1]))
            qr_codes.append((dec_inf, center_x, center_y, point))
    return qr_codes

# -----------------------------------------------------------
# function to annotate image with QR codes
# -----------------------------------------------------------
def annotate_image(img, qr_codes):
    for dec_inf, center_x, center_y, point in qr_codes:
        try:
            dec_inf_shiftjis = dec_inf.encode('latin1').decode('shift_jis')
        except UnicodeDecodeError:
            dec_inf_shiftjis = dec_inf
        print('QRコードの内容:', dec_inf_shiftjis)
        print(f'QRコードの中心座標: ({center_x}, {center_y})')
        x = point[0][0]
        y = point[0][1]
        img = cv2.putText(img, dec_inf_shiftjis, (x, y-6), font, .6, (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.polylines(img, [point], True, (0, 255, 0), 2, cv2.LINE_AA)
        img = cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
    return img

# -----------------------------------------------------------
# function to resize and display image
# -----------------------------------------------------------
def resize_and_display(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized Image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------------------------------------
# sample program
# -----------------------------------------------------------
img_BGR = cv2.imread(FILE_PNG_AB, cv2.IMREAD_COLOR)
if img_BGR is not None:
    resized_img_BGR = compress_image(img_BGR, SCALE_PERCENT)
    all_qr_codes = []
    for angle in ROTATION_ANGLES:
        rotated_img = rotate_image(resized_img_BGR, angle)
        qr_codes = function_qrdec_cv2(rotated_img)
        all_qr_codes.extend(qr_codes)
    # Average positions of QR codes if the same QR code is detected multiple times
    unique_qr_codes = {}
    for dec_inf, center_x, center_y, point in all_qr_codes:
        if dec_inf in unique_qr_codes:
            unique_qr_codes[dec_inf].append((center_x, center_y, point))
        else:
            unique_qr_codes[dec_inf] = [(center_x, center_y, point)]
    final_qr_codes = []
    for dec_inf, values in unique_qr_codes.items():
        avg_x = int(np.mean([v[0] for v in values]))
        avg_y = int(np.mean([v[1] for v in values]))
        point = values[0][2]  # Use points from the first occurrence
        final_qr_codes.append((dec_inf, avg_x, avg_y, point))
    # Annotate the image with final QR code positions
    annotated_img = annotate_image(resized_img_BGR, final_qr_codes)
    # 最終画像を縮小表示
    resize_and_display(annotated_img, SHOW_SCALE_PERCENT)
else:
    print(f"Error: Unable to read image {FILE_PNG_AB}")
