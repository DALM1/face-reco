import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QPushButton, QFileDialog, QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class FaceDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("facetrack")
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(3, 3, int(screen.width() * 0.5), int(screen.height() * 0.5))

        self.main_layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.main_layout.addWidget(self.video_label)

        self.summary_label = QLabel("Nombre de visages détectés 0", self)
        self.main_layout.addWidget(self.summary_label)

        self.face_carousel = QHBoxLayout()
        self.face_carousel.setSpacing(5)

        self.scroll_area = QScrollArea(self)
        self.scroll_area_widget = QWidget()
        self.scroll_area_widget.setLayout(self.face_carousel)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.download_button = QPushButton("Télécharger les images")
        self.download_button.clicked.connect(self.save_images)
        self.main_layout.addWidget(self.download_button)

        self.setLayout(self.main_layout)

        self.capture = cv2.VideoCapture(1)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.detected_faces = []
        self.face_widgets = []
        self.dialogs = []

    def update_frame(self):
        ret, frame = self.capture.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_img = frame[y:y + h, x:x + w]

                if not self.is_face_already_detected(face_img):
                    self.detected_faces.append(face_img)
                    self.add_face_to_carousel(face_img)

            self.summary_label.setText(f"Nombre de visages détectés {len(self.detected_faces)}")

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def is_face_already_detected(self, new_face):
        for face in self.detected_faces:
            if self.are_faces_similar(face, new_face):
                return True
        return False

    def are_faces_similar(self, face1, face2):
        face1_resized = cv2.resize(face1, (50, 50))
        face2_resized = cv2.resize(face2, (50, 50))
        difference = cv2.absdiff(face1_resized, face2_resized)
        return cv2.mean(difference)[0] < 20

    def add_face_to_carousel(self, face_img):
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        height, width, channel = face_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(face_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        face_label = QLabel(self)
        face_label.setPixmap(QPixmap.fromImage(q_img).scaled(100, 100))

        face_label.mousePressEvent = lambda event, img=face_img: self.show_large_face(img)

        self.face_carousel.addWidget(face_label)
        self.face_widgets.append(face_label)

    def show_large_face(self, face_img):
        dialog = QDialog(self)
        dialog.setWindowTitle("Visage détecté")
        dialog.setFixedSize(400, 400)

        layout = QVBoxLayout()

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        height, width, channel = face_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(face_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        face_label = QLabel(dialog)
        face_label.setPixmap(QPixmap.fromImage(q_img).scaled(300, 300))

        layout.addWidget(face_label)

        download_button = QPushButton("Télécharger ce visage", dialog)
        download_button.clicked.connect(lambda: self.save_single_image(face_img))
        layout.addWidget(download_button)

        dialog.setLayout(layout)
        self.dialogs.append(dialog)
        dialog.show()

    def save_single_image(self, face_img):
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer l'image", "", "JPEG Files (*.jpg);;PNG Files (*.png)")
        if file_path:
            cv2.imwrite(file_path, face_img)
            print(f"Image enregistrée {file_path}")

    def save_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisissez un dossier")

        if folder:
            for i, face in enumerate(self.detected_faces):
                face_path = os.path.join(folder, f"face_{i + 1}.jpg")
                cv2.imwrite(face_path, face)
            print(f"Images enregistrées dans {folder}")

    def closeEvent(self, event):
        self.capture.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
