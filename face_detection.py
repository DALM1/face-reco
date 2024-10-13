import sys
import cv2
import os
import hashlib
import numpy as np
import pyperclip
from cryptography.fernet import Fernet
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QPushButton, QFileDialog, QDialog, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FaceDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CyberTrack")
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(3, 3, int(screen.width() * 0.5), int(screen.height() * 0.5))

        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

        self.setStyleSheet("background-color: #2e2e2e; color: #ffffff;")
        self.main_layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.main_layout.addWidget(self.video_label)

        self.summary_label = QLabel("Visages détectés 0", self)
        self.main_layout.addWidget(self.summary_label)

        self.target_input = QLineEdit(self)
        self.target_input.setPlaceholderText("Hash du visage cible (SHA256)")
        self.main_layout.addWidget(self.target_input)

        self.face_carousel = QHBoxLayout()
        self.face_carousel.setSpacing(3)

        self.scroll_area = QScrollArea(self)
        self.scroll_area_widget = QWidget()
        self.scroll_area_widget.setLayout(self.face_carousel)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.upload_button = QPushButton("Charger une photo pour la recherche")
        self.upload_button.clicked.connect(self.upload_and_search_face)
        self.main_layout.addWidget(self.upload_button)

        self.download_button = QPushButton("Télécharger les images cryptées")
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
        self.target_face_hash = None

        self.face_data = []
        self.face_labels = []
        self.model = None

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

                face_hash = self.hash_face(face_img)
                if self.target_face_hash and face_hash == self.target_face_hash:
                    self.alert_detected_face()

            self.summary_label.setText(f"Visages détectés {len(self.detected_faces)}")

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.video_label.setPixmap(QPixmap.fromImage(q_img))

        if self.target_input.text():
            self.target_face_hash = self.target_input.text()

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

    def hash_face(self, face_img):
        resized_face = cv2.resize(face_img, (100, 100))
        face_bytes = resized_face.tobytes()
        face_hash = hashlib.sha256(face_bytes).hexdigest()
        return face_hash

    def add_face_to_carousel(self, face_img):
        face_hash = self.hash_face(face_img)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_rgb_with_text = self.add_hash_to_face(face_rgb, face_hash)

        height, width, channel = face_rgb_with_text.shape
        bytes_per_line = 3 * width
        q_img = QImage(face_rgb_with_text.data, width, height, bytes_per_line, QImage.Format_RGB888)

        face_label = QLabel(self)
        face_label.setPixmap(QPixmap.fromImage(q_img).scaled(100, 100))

        face_label.mousePressEvent = lambda event, img=face_img, h=face_hash: self.show_large_face(img, h)

        self.face_carousel.addWidget(face_label)
        self.face_widgets.append(face_label)

    def add_hash_to_face(self, face_img, face_hash):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_color = (255, 255, 255)
        thickness = 1
        x, y = 10, 90
        face_img_with_text = face_img.copy()
        cv2.putText(face_img_with_text, face_hash[:10], (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        return face_img_with_text

    def show_large_face(self, face_img, face_hash):
        dialog = QDialog(self)
        dialog.setWindowTitle("Visage détecté")
        dialog.setFixedSize(400, 500)

        layout = QVBoxLayout()

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        height, width, channel = face_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(face_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        face_label = QLabel(dialog)
        face_label.setPixmap(QPixmap.fromImage(q_img).scaled(300, 300))
        layout.addWidget(face_label)

        hash_label = QLabel(f"HASH {face_hash}", dialog)
        layout.addWidget(hash_label)

        copy_button = QPushButton("Copier le hash", dialog)
        copy_button.clicked.connect(lambda: self.copy_to_clipboard(face_hash))
        layout.addWidget(copy_button)

        download_button = QPushButton("Télécharger ce visage", dialog)
        download_button.clicked.connect(lambda: self.save_single_image(face_img))
        layout.addWidget(download_button)

        dialog.setLayout(layout)
        self.dialogs.append(dialog)
        dialog.show()

    def copy_to_clipboard(self, face_hash):
        pyperclip.copy(face_hash)

    def upload_and_search_face(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Charger une photo", "", "Images Files (*.jpg *.png)")
        if file_path:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = img[y:y + h, x:x + w]
                face_hash = self.hash_face(face_img)

                if face_hash in [self.hash_face(face) for face in self.detected_faces]:
                    self.alert_detected_face()

    def save_single_image(self, face_img):
        encrypted_img = self.encrypt_image(face_img)
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer l'image cryptée", "", "Encrypted Files (*.enc)")
        if file_path:
            with open(file_path, 'wb') as f:
                f.write(encrypted_img)

    def encrypt_image(self, face_img):
        face_bytes = face_img.tobytes()
        encrypted_data = self.cipher_suite.encrypt(face_bytes)
        return encrypted_data

    def save_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisissez un dossier")

        if folder:
            for i, face in enumerate(self.detected_faces):
                encrypted_img = self.encrypt_image(face)
                face_path = os.path.join(folder, f"face_{i + 1}.enc")
                with open(face_path, 'wb') as f:
                    f.write(encrypted_img)
            print(f"Images cryptées enregistrées dans {folder}")

    def alert_detected_face(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("ALERTE <Visage Cible>")
        layout = QVBoxLayout()
        alert_label = QLabel("Visage cible détecté")
        layout.addWidget(alert_label)
        dialog.setLayout(layout)
        dialog.exec_()

    def closeEvent(self, event):
        self.capture.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())
