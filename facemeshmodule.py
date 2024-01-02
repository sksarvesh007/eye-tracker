import cv2
import mediapipe as mp
import time 

class FaceMeshModule:
    def __init__(self , staticMode=False, maxFaces=2):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        # self.minDetectionCon = minDetectionCon
        # self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        
        self.mpFaceMesh = mp.solutions.face_mesh 
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img , draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLMS in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLMS, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
            for faceLMS in results.multi_face_landmarks:
                for id, lm in enumerate(faceLMS.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
        return img
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    face_mesh_module = FaceMeshModule()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = face_mesh_module.findFaceMesh(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS : {int(fps)}', (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 3)
        cv2.imshow("IMAGE", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
if __name__ == "__main__":
    main()
cv2.destroyAllWindows()

