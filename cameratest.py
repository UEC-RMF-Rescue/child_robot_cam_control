import requests # type: ignore

# ESP32-CAMのIPアドレス
esp32_cam_ip = "http://192.168.179.12/capture"

# 写真を取得
response = requests.get(esp32_cam_ip)

if response.status_code == 200:
    with open("photo.jpg", "wb") as f:
        f.write(response.content)
    print("Photo saved as photo.jpg")
else:
    print("Failed to get photo")
