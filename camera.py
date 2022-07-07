import cv2
import time
from utills import mask_detection

camera = 0
if camera == 0:
	path = 0
else:
	path = 1


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(path)
		self.pTime = 0

	def __del__(self):
		self.video.release()

	def get_frame(self):
		try:
			ret, image = self.video.read()

			image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
			frame = mask_detection(image)
			cTime = time.time()
			fps = 1.0 / (cTime - self.pTime)
			txt = 'FPS: ' + str(int(fps))
			cv2.rectangle(frame, (0, 0), (70, 20), (107, 106, 41), cv2.FILLED)
			cv2.putText(frame, txt, (4, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
			self.pTime = cTime
			# cv2.imshow('web-cam', frame)
			ret, jpeg = cv2.imencode('.jpg', frame)
			return jpeg.tobytes()
		finally:
			pass





