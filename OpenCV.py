#import libraryที่จำเป็น
#นำเข้า Library OpenCV ใช้สำหรับประมวลผลภาพ, เปิดกล้อง, วาดกราฟิก, และโหลด AI Model



import numpy as np 
import cv2 

#รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]


#สีตัวกรอบที่วาดrandomใหม่ทุกครั้ง
#สร้างสีแบบสุ่ม 21 สี (ตามจำนวน Class) แต่ละสีมี 3 ค่า (B, G, R)
#ทำให้แต่ละประเภทวัตถุมีกรอบสีต่างกัน และสีจะสุ่มใหม่ทุกครั้งที่รันโปรแกรม

COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))
#โหลดmodelจากแฟ้ม
""""โหลด AI Model จากไฟล์ 2 ไฟล์
.prototxt = ไฟล์กำหนด โครงสร้าง ของ Neural Network
.caffemodel = ไฟล์เก็บ น้ำหนัก (Weights) ที่ฝึกมาแล้ว"""

net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")


#เลือกวิดีโอ/เปิดกล้อง
#เปิดกล้อง Webcam ตัวที่ 0 (กล้องหลักของเครื่อง) หากต้องการใช้วิดีโอไฟล์ เปลี่ยน 0 เป็น "video.mp4"

cap = cv2.VideoCapture(0)

#วนลูปไม่สิ้นสุด เพื่อประมวลผลภาพจากกล้องทีละเฟรมอย่างต่อเนื่อง
while True:
	#เริ่มอ่านในแต่ละเฟรม
	ret, frame = cap.read()
	if ret:
		(h,w) = frame.shape[:2]
		#ทำpreprocessing
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
		net.setInput(blob)
		#feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			percent = detections[0,0,i,2]
			#กรองเอาเฉพาะค่าpercentที่สูงกว่า0.5 เพิ่มลดได้ตามต้องการ
			if percent > 0.5:
				class_index = int(detections[0,0,i,1])
				box = detections[0,0,i,3:7]*np.array([w,h,w,h])
				(startX, startY, endX, endY) = box.astype("int")

				#ส่วนตกแต่งสามารถลองแก้กันได้ วาดกรอบและชื่อ
				label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
				cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
				y = startY - 15 if startY-15>15 else startY+15
				cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

		cv2.imshow("Frame", frame)
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break

#หลังเลิกใช้แล้วเคลียร์memoryและปิดกล้อง
cap.release()
cv2.destroyAllWindows()
