#====== peripheralUI.py ======#
# Common UI features that may be used in other modules,
# but are not standalone. Typically be populated with
# content when initialised. e.g. popup info boxes.

#====== Libraries ======#
from PyQt5.QtWidgets import *
import PyQt5.QtGui as QtGui;
import Model.model_training as model_training
import resources as r;
from torchvision import datasets, transforms;
import urllib.request


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

class createModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle("Train Model")
        self.textBox = QTextEdit()
        self.textBox.setReadOnly(True)

        downloadDataButton = QPushButton("&Download Dataset", self);
        downloadDataButton.setToolTip("Download MNIST Dataset");
        downloadDataButton.setStatusTip("Download MNIST Dataset");
        downloadDataButton.clicked.connect(self.downloadMNISTData);

        self.progressBar = QProgressBar(self)
        self.completed = 0

        trainButton = QPushButton("&Train Model", self);
        trainButton.setToolTip("Train Model");
        trainButton.setStatusTip("Train Model");
        trainButton.clicked.connect(self.trainModel)

        cancelButton = QPushButton("&Close", self);
        cancelButton.setToolTip("Close model-creation window");
        cancelButton.setStatusTip("Close model-creation window");
        cancelButton.clicked.connect(self.close)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.textBox)
        self.layout.addWidget(self.progressBar)
        self.layout.addWidget(downloadDataButton)
        self.layout.addWidget(trainButton)
        self.layout.addWidget(cancelButton)
        self.setLayout(self.layout)

    def downloadMNISTData(self):
        self.textBox.append("Downloading dataset...")
        self.textBox.repaint()

        # Downloading MNIST Dataset (if it doesn't already exist)
        try:
            datasets.MNIST(root= '',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
            self.textBox.append("Dataset already downloaded!")
        except:
            datasets.MNIST(root= '',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
            self.textBox.append("Dataset downloaded!")
        finally:
            self.completed = 50
            self.progressBar.setValue(self.completed)

    def trainModel(self):
        self.textBox.append("Training...")
        try:
            accuracy = model_training.trainRecognitionModel(self.completed, self.progressBar)
            self.textBox.append("Training Done\nAccuracy: " + str(round(float(accuracy))) + "%")
        except Exception as e:
            self.textBox.append("Error training the model. Make sure the model has been downloaded first by pressing the 'Download Dataset' button")
            print(e)


