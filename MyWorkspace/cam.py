import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
# 很多相机必须强制V4L2后端，默认后端经常出线程问题
if not cap.isOpened():
    print("无法打开摄像头")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取失败")
            break
        cv2.imshow("cam", frame)
        if cv2.waitKey(1) == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
