from vidstab.VidStab import VidStab
import cv2

stabilizer = VidStab()
vidcap = cv2.VideoCapture('/home/dilan/Desktop/Final Year Project/Programming Testing/Filter Test/Videos/cleantrafficvideosar.mp4')

while True:
     grabbed_frame, frame = vidcap.read()

     if frame is None:
        continue

     # Pass frame to stabilizer even if frame is None
     # stabilized_frame will be an all black frame until iteration 30
     stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,
                                                   smoothing_window=30)
     if stabilized_frame is None:
         # There are no more frames available to stabilize
         break

     cv2.imshow('test', stabilized_frame)
     # Press Q on keyboard to  exit
     if cv2.waitKey(25) & 0xFF == ord('q'):
         break

