#====== peripheralUI.py ======#
# Common UI features that may be used in other modules,
# but are not standalone. Typically be populated with
# content when initialised. e.g. popup info boxes.

#====== Libraries ======#
import resources as r
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from torchvision import datasets, transforms;
import gzip
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# import Model.model_training as model_linear_training
# #import Model.model1 as model_temp_training
# MODEL_LIST = ["Linear", "Convolutional", "Complex"]


#====== Code ======#

# PopupBox()
# Basic low priority box that displays rich-format text in a box,
# and can be closed when the user chooses. Independent window.

class PopupBox(QWidget):
    # Text box that pops up in a new windows.
    # Useful for displaying reports and detailed information.
    
    def __init__(self, title="Popup Box", icon=None):
        super().__init__();
        self.initPopup(title, icon);

    def initPopup(self, title, icon):
        self.setWindowTitle(title);
        self.setWindowIcon(QIcon(icon));
        self.setGeometry(400, 300, 400, 400);
        
        self.button = QPushButton("Close", self);
        self.button.clicked.connect(self.exitPopup);
        self.browser = QTextBrowser();
        self.browser.setAcceptRichText(True);
        self.browser.setOpenExternalLinks(True);

        self.vbox = QVBoxLayout();
        self.vbox.addWidget(self.browser);
        self.vbox.addWidget(self.button);

        self.setLayout(self.vbox);
        self.show();

    def assignText(self, message):
        self.browser.append(message);

    def exitPopup(self):
        self.close();


# ErrorBox()
# Simple warning info box with configurable parameters.
# ErrorBox is blocking; the user must close it to continue
# to use the app.

class ErrorBox(QMessageBox):
    def __init__(self, title="Error", msg="Error", detail=""):
        super().__init__();
        self.init(title, msg, detail);

    def init(self, title, msg, detail):
        self.setWindowTitle(title);
        self.setWindowIcon(QIcon(r.ICON_WARNING));
        self.setIcon(QMessageBox.Warning);
        
        self.setText(msg);
        detail = str(detail);       # must convert to string for readable form
        if (isinstance(detail, str)):
            self.setInformativeText(detail);

    def render(self):
        # Enable blocking (cannot use app until box acknowledged):
        self.exec_();
        self.show();


# Download and Training Dialogue
# Allows the user to download the MNIST dataset and train the model.
# Shows progress downloading and training with a progress bar.

# class CreateModelDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent=parent)

#         self.setWindowTitle("Train Model")
#         self.setWindowIcon(QIcon(r.ICON_WORKING))
        
#         self.textBox = QTextEdit()
#         self.textBox.setReadOnly(True)

#         self.progressBar = QProgressBar(self)
#         self.completed = 0

#         self.downloadDataButton = QPushButton("&Download Dataset", self);
#         self.downloadDataButton.setToolTip("Download MNIST Dataset");
#         self.downloadDataButton.setStatusTip("Download MNIST Dataset");
#         self.downloadDataButton.clicked.connect(self.downloadMNISTData);

#         self.modelCombo = QComboBox();
#         self.modelCombo.setToolTip("Select your model structure")
#         self.modelCombo.addItems(MODEL_LIST);

#         self.trainButton = QPushButton("&Train Model", self);
#         self.trainButton.setToolTip("Train Model");
#         self.trainButton.setStatusTip("Train Model");
#         self.trainButton.clicked.connect(lambda startTrain: self.trainModel(self.modelCombo.currentText()));
#         # We must register an interim function ('startTrain()') for the event to call a func with params correctly
#         # https://forum.qt.io/topic/60640/pyqt-immediately-calling-function-handler-upon-setup-why/4

#         self.cancelButton = QPushButton("&Close", self);
#         self.cancelButton.setToolTip("Close model-creation window");
#         self.cancelButton.setStatusTip("Close model-creation window");
#         self.cancelButton.clicked.connect(self.close)

#         self.layout = QVBoxLayout()
#         self.layout.addWidget(self.textBox)
#         self.layout.addWidget(self.progressBar)
#         self.layout.addWidget(self.downloadDataButton)
#         self.layout.addWidget(self.modelCombo)
#         self.layout.addWidget(self.trainButton)
#         self.layout.addWidget(self.cancelButton)
#         self.setLayout(self.layout)


#     def downloadMNISTData(self):
#         self.textBox.append("Downloading dataset...")
#         self.textBox.repaint()

#         # Downloading MNIST Dataset (if it doesn't already exist)
#         try:
#             datasets.MNIST(root= r.MODULE_DIR,
#                             train=True,
#                             transform=transforms.ToTensor(),
#                             download=True)
#             self.textBox.append("Dataset already downloaded!")
#         except:
#             datasets.MNIST(root= r.MODULE_DIR,
#                             train=True,
#                             transform=transforms.ToTensor(),
#                             download=True)
#             self.textBox.append("Dataset downloaded!")
#         finally:
#             self.completed = 50
#             self.progressBar.setValue(self.completed)

#     #Method to start training the model
#     def trainModel(self, model_str):
#         self.textBox.append(f"Training {model_str}...");
#         try:
#             if (model_str == "Convolutional"):
#                 accuracy = 0;
#                 # accuracy = model_..._training.trainRecognitionModel(self.completed, self.progressBar);
#             elif (model_str == "Complex"):
#                 pass;
#             else:
#                 # default to linear model
#                 accuracy = model_linear_training.trainRecognitionModel(self.completed, self.progressBar);

#         except Exception as e:
#             self.textBox.append("Error training the model. Make sure the model has been downloaded first by pressing the 'Download Dataset' button");
#             print(e);
#         else:
#             self.textBox.append("Training Done\nAccuracy: " + str(round(float(accuracy))) + "%");


# Image Viewer Dialogue
# View images in a dataset one-by-one. Change image with 'next' and 'previous'.

class ViewImagesDlg(QDialog):
    def __init__(self, datasetType):
        super().__init__()
        self.setWindowTitle("Dataset Viewer")

        #Determining which dataset to view
        if datasetType == "training":
            self.filePath = 'MNIST/raw/train-images-idx3-ubyte.gz'
            self.num_images = 60000
        else:
            self.filePath = 'MNIST/raw/t10k-images-idx3-ubyte.gz'
            self.num_images = 10000

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        #Decompressing the .gz file and appending its contents to a list
        self.imgList = self.generateImgList()
        self.imageIndex = 0

        #Adding image as a widget
        self.img = QLabel(self)
        self.img.setPixmap(self.convertToPixmap(self.imgList[self.imageIndex]))
        self.layout.addWidget(self.img)

        #Button to show next image
        nextButton = QPushButton("&Next", self)
        nextButton.clicked.connect(self.showNextImg)
        self.layout.addWidget(nextButton)

        #Button to show previous image
        previousButton = QPushButton("&Previous", self)
        previousButton.clicked.connect(self.showPrevImg)
        self.layout.addWidget(previousButton)


    def generateImgList(self):
        f = gzip.open(self.filePath,'r')

        image_size = 28

        #Skip first 16 bytes as they're not pixels, according to: http://yann.lecun.com/exdb/mnist/
        f.read(16)

        buf = f.read(image_size * image_size * self.num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(self.num_images, image_size, image_size, 1)

        return data

    #Method to show next image when next button is clicked
    def showNextImg(self):
        self.imageIndex += 1
        if self.imageIndex > self.num_images - 1:
            self.imageIndex = 0
        print(self.imageIndex)
        self.img.setPixmap(self.convertToPixmap(self.imgList[self.imageIndex]))
    
    #Method to show previous image when previous button is clicked
    def showPrevImg(self):
        self.imageIndex -= 1
        if self.imageIndex < 0:
            self.imageIndex = self.num_images - 1
        print(self.imageIndex)
        self.img.setPixmap(self.convertToPixmap(self.imgList[self.imageIndex]))

    #Converting directly to pixmap distorts the image, therefore we save it first before reading it as a cv2 img
    def convertToPixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        cv2.imwrite("img.png", img)
        img = cv2.imread("img.png")
        os.remove("img.png")
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap(qImg).scaled(150, 150, Qt.KeepAspectRatio)

