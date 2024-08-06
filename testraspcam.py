import socket
import numpy as np
import cv2
import os

HOST = "your IP adress"  # Raspberry Pi IP address
PORT = 5569
SAVE_DIR = r"your file"  # Directory to save images

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def getimage():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))

        buf = bytearray()
        recvlen = 1024 * 8

        while True:
            receivedstr = sock.recv(recvlen)
            if not receivedstr:
                break
            buf.extend(receivedstr)

        sock.close()
        recdata = np.frombuffer(buf, dtype='uint8')
        return cv2.imdecode(recdata, 1)
    except ConnectionRefusedError as e:
        print(f"Connection failed: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

saved_count = 0
max_saves = 10

while saved_count < max_saves:
    img = getimage()
    
    if img is not None:
        cv2.imshow('Capture', img)
        
        key = cv2.waitKey(5)
        if key != -1:  # If any key is pressed
            saved_count += 1
            filename = os.path.join(SAVE_DIR, f'Capture{saved_count}.jpg')
            cv2.imwrite(filename, img)
            print(f"Saved {filename}")
    else:
        print("Failed to receive image")
        cv2.waitKey(1000)  # Wait longer before retrying if failed

cv2.destroyAllWindows()
print("Image capture complete")
