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
        #Determining which dataset to view
        if datasetType == "training":
            self.filePath = 'MNIST/raw/train-images-idx3-ubyte.gz'
            self.num_images = 60000
        else:
            self.filePath = 'MNIST/raw/t10k-images-idx3-ubyte.gz'
            self.num_images = 10000

        #Decompressing the .gz file and appending its contents to a list
        self.imgList = self.generateImgList()
        self.imageIndex = 0

        # Layout
        self.vbox = QVBoxLayout();
        self.btnbox = QHBoxLayout();
        self.btnbox_widget = QWidget();
        self.btnbox_widget.setLayout(self.btnbox);

        self.gallery = QGridLayout();
        self.gallery_widget = QWidget();
        self.gallery_widget.setLayout(self.gallery);

        self.gallery_scroll = QScrollArea();
        self.gallery_scroll.setWidget(self.gallery_widget);

        self.setLayout(self.vbox);
        self.vbox.addWidget(self.gallery_scroll);
        self.vbox.addWidget(self.btnbox_widget);

        # -- Gallery
        self.GRID_X = 5;
        self.GRID_Y = 10;
        self.imageIndex = self.populate(self.imageIndex);
        self.gallery_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn);
        self.gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn);
        self.gallery_scroll.setWidgetResizable(True);
        # -- Button to show previous image
        self.previousButton = QPushButton("&Previous", self);
        self.previousButton.clicked.connect(self.showPrevImg);
        self.btnbox.addWidget(self.previousButton);
        # -- Label to show progress
        self.progressLabel = QLabel();
        self.progressLabel.setAlignment(Qt.AlignHCenter);
        self.updateProgressLabel();
        self.btnbox.addWidget(self.progressLabel);
        # -- Button to show next image
        self.nextButton = QPushButton("&Next", self);
        self.nextButton.clicked.connect(self.showNextImg);
        self.btnbox.addWidget(self.nextButton);

        # Window Layout
        self.setWindowTitle("Dataset Viewer")
        self.setWindowIcon(QIcon(r.ICON_FIND));
        self.setGeometry(200, 200, (120 + 28*5* self.GRID_X), (120 + 28*5*5));

    def updateProgressLabel(self):
        self.progressLabel.setText(f"{self.imageIndex} / {self.num_images}");

    def populate(self, index):
        # Generate positions tuple
        positions = [(x, y) for x in range(self.GRID_X) for y in range(self.GRID_Y)]
        
        # Add images to the grid:
        for x, y in positions:
            if index < self.num_images:
                self.gallery.addWidget(self.generateImage(index), y, x);
                index += 1;

        return index;

    def generateImage(self, index):
        img = QLabel();
        img.setPixmap(self.convertToPixmap(self.imgList[index]));
        return img;


    def generateImgList(self):
        f = gzip.open(self.filePath,'r')

        image_size = 28

        #Skip first 16 bytes as they're not pixels, according to: http://yann.lecun.com/exdb/mnist/
        f.read(16)

        buf = f.read(image_size * image_size * self.num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(self.num_images, image_size, image_size, 1)

        return data

    #Method to show next images when next button is clicked
    def showNextImg(self):
        self.imageIndex = self.populate(self.imageIndex)

        self.updateProgressLabel();

    #Method to show previous images when previous button is clicked
    def showPrevImg(self):
        temp = self.imageIndex - (2 * self.GRID_X * self.GRID_Y);
        if (temp < 0): temp = 0;
        num_max = self.num_images - 1 - (self.GRID_X * self.GRID_Y);
        if (temp >= num_max): temp = num_max;
        self.imageIndex = self.populate(temp)

        self.updateProgressLabel();

    #Converting directly to pixmap distorts the image, therefore we save it first before reading it as a cv2 img
    def convertToPixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        cv2.imwrite("img.png", img)
        img = cv2.imread("img.png")
        os.remove("img.png")
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap(qImg).scaled(150, 150, Qt.KeepAspectRatio)

