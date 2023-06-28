# Класс оконного приложения

import cv2
import qimage2ndarray
import torch
from PyQt6 import QtWidgets, QtGui
from sentence_transformers import SentenceTransformer

from generation_network import Generator
from main_window_ui import Ui_MainWindow as MainWindowUI


class MainWindow(QtWidgets.QMainWindow, MainWindowUI):
    def __init__(self):
        super().__init__(None)
        self.setupUi(self)
        self.generators = self.load_generators()
        self.image = None
        self.generateButton.clicked.connect(self.generate_button_pressed)
        self.saveButton.clicked.connect(self.save_button_pressed)
        self.show()

    def load_generators(self):
        generators = {}
        generator_flowers = Generator(in_embedding_dim=768,
                                      projected_embedding_dim=128,
                                      noise_dim=100,
                                      channels_dim=64,
                                      out_channels=3)
        try:
            generator_flowers.load_state_dict(torch.load('generators/generator_flowers.pth'))
        except FileNotFoundError:
            pass
        generators[0] = generator_flowers

        generator_birds = Generator(in_embedding_dim=768,
                                    projected_embedding_dim=128,
                                    noise_dim=100,
                                    channels_dim=64,
                                    out_channels=3)
        try:
            generator_birds.load_state_dict(torch.load('generators/generator_birds.pth'))
        except FileNotFoundError:
            pass
        generators[1] = generator_birds
        return generators

    def generate_button_pressed(self):
        encoder = SentenceTransformer('all-mpnet-base-v2', device='cpu')
        text = self.requestLineEdit.text()
        embedding = encoder.encode([text], convert_to_tensor=True)
        noise = torch.randn(1, 100, 1, 1)
        generator = self.generators[self.generatorTypeSlider.value()]
        generator.eval()
        generator.to('cpu')
        fake_image = generator(embedding.to('cpu'), noise.to('cpu'))[0]

        fake_image = fake_image.permute(1, 2, 0).detach().numpy()
        fake_image = (fake_image + 1) / 2
        fake_image = cv2.resize(fake_image, dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)
        fake_image *= 255

        self.image = qimage2ndarray.array2qimage(fake_image)
        pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(self.image))
        self.imageLabel.setPixmap(pixmap)

    def save_button_pressed(self):
        if self.image is None:
            return

        dialog = QtWidgets.QFileDialog()
        options = dialog.options()
        options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save image', '', 'Image files (*.png)', options=options
        )
        if file_name[-4:] != '.png':
            file_name += '.png'
        self.image.save(file_name, format='png')
