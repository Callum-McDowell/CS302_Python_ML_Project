#====== modelManager.py ======#
# Trains the models and selects which one to use.
# All model and prediction requests must be sent to
# this module.

#====== Libraries ======#
import resources as r
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from torchvision import datasets, transforms;
import matplotlib.pyplot as plt
import numpy as np
import peripheralUI

import Model.Model_Linear.model_linear as model_linear
import Model.Model_Linear.model_linear_prediction as model_linear_prediction
# ...
MODEL_LIST = ["Linear", "Convolutional", "Complex"]
# Note: default to Linear


# modelManager
# Interface for requests to the model
# Use it to select the current model, train

class modelManager():
    def __init__(self):
        self.model_name = MODEL_LIST[0];
        self.model_weights_dir = None;
        self.plot_probabilities = None;

    def setModelName(self, name : str):
        if (isinstance(name, str)):
            self.model_name = name;

    def changeModelWeightsDir(self, owner):
        # Owner is QWidget to act as parent
        weights_dir, _ = QFileDialog.getOpenFileName(owner,"Please select model weights", "Model/","pickle files (*.pkl)")
        if (len(weights_dir) > 0):
            self.model_weights_dir = weights_dir;

    def predictWithModel(self, image):
        try:
            if (self.model_name == "Convolutional"):
                pass;
                # self.pred, self.plt = predFile.predict(image, self.model_weights_dir);
                # return self.pred, self.plt;
            elif (self.model_name == "Complex"):
                pass;
                # self.pred, self.plt = predFile.predict(image, self.model_weights_dir);
                # return self.pred, self.plt;
            else:
                pred, self.plot_probabilities = model_linear_prediction.predict(image, self.model_weights_dir);
                
            plot = self.createBarPlot();    
            return pred, plot;
        except Exception as e:
            # If an invalid file is loaded...
            self.generateErrorBox("Error", "Invalid Model", e)
            return;

    def createBarPlot(self):
        plot = self.plot_bar(self.plot_probabilities);
        plt.show();

        mngr = plt.get_current_fig_manager();
        mngr.window.setGeometry(50,100, 600,600);
        return plot;

    def plot_bar(self, probability):
        plt.close() # Close previous plot if it's still open

        # Normalise to 1 to get % values
        temp = [(100 * float(i))/sum(probability) for i in probability];
        probability = temp;

        # Get array of indices 
        index = np.arange(len(probability)) 
        # Plot index on x-axis and probability on y-axis
        plot = plt.bar(index, probability)

        #Add labels
        plt.xlabel('Digit', fontsize=15)
        plt.ylabel('Probability', fontsize=20)
        plt.xticks(index, fontsize=8, rotation=30)
        plt.title('Model Prediction Probability')
        plt.show()
        return plot;

    def generateErrorBox(self, title="Error", message="Error", detail="None"):
        error_box = peripheralUI.ErrorBox(title, message, detail);
        error_box.render();



# Download and Training Dialogue
# Allows the user to download the MNIST dataset and train the model.
# Shows progress downloading and training with a progress bar.

class CreateModelDialog(QDialog):
    def __init__(self, parent=None, manager=None):
        # parent is the QWidget that will own the dialog
        # manager is the modelManager() instance that stores the model data
        super().__init__(parent=parent)
        self.model_manager = manager;

        self.setWindowTitle("Train Model")
        self.setWindowIcon(QIcon(r.ICON_WORKING))
        
        self.textBox = QTextEdit()
        self.textBox.setReadOnly(True)

        self.progressBar = QProgressBar()

        self.downloadDataButton = QPushButton("&Download Dataset", self);
        self.downloadDataButton.setToolTip("Download MNIST Dataset");
        self.downloadDataButton.setStatusTip("Download MNIST Dataset");
        self.downloadDataButton.clicked.connect(self.downloadMNISTData);

        self.modelCombo = QComboBox();
        self.modelCombo.setToolTip("Select your model structure")
        self.modelCombo.addItems(MODEL_LIST);

        self.trainButton = QPushButton("&Train Model", self);
        self.trainButton.setToolTip("Train Model");
        self.trainButton.setStatusTip("Train Model");
        self.trainButton.clicked.connect(lambda startTrain: self.setAndTrainModel(self.modelCombo.currentText()));
        # We must register an interim function ('startTrain()') for the event to call a func with params correctly
        # https://forum.qt.io/topic/60640/pyqt-immediately-calling-function-handler-upon-setup-why/4

        self.cancelButton = QPushButton("&Close", self);
        self.cancelButton.setToolTip("Close model-creation window");
        self.cancelButton.setStatusTip("Close model-creation window");
        self.cancelButton.clicked.connect(self.close)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.textBox)
        self.layout.addWidget(self.progressBar)
        self.layout.addWidget(self.downloadDataButton)
        self.layout.addWidget(self.modelCombo)
        self.layout.addWidget(self.trainButton)
        self.layout.addWidget(self.cancelButton)
        self.setLayout(self.layout)

        self.exec_();


    def downloadMNISTData(self):
        self.textBox.append("Downloading dataset...")
        self.textBox.repaint()

        # Downloading MNIST Dataset (if it doesn't already exist)
        try:
            datasets.MNIST(root= r.MODULE_DIR,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
            self.textBox.append("Dataset already downloaded!")
        except:
            datasets.MNIST(root= r.MODULE_DIR,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
            self.textBox.append("Dataset downloaded!")
        finally:
            self.progressBar.setValue(50)

    def setAndTrainModel(self, model_str):
        self.model_manager.setModelName(model_str);
        self.trainModel(model_str);

    def trainModel(self, model_str):
        self.downloadMNISTData();

        self.textBox.append(f"Training {model_str} model...");
        try:
            if (model_str == "Convolutional"):
                self.accuracy = 0;
                # accuracy = model_..._training.trainRecognitionModel(self.completed, self.progressBar);
            elif (model_str == "Complex"):
                self.accuracy = 0;
            else:
                # default to linear model
                self.accuracy = 0;

                x = model_linear.modelTrainingFramework();
                self.accuracy = x.trainModel(self.progressBar);

                #self.accuracy = model_linear.trainModel();
                # accuracy = model_linear.trainRecognitionModel(self.completed, self.progressBar);

        except Exception as e:
            self.textBox.append("Error training the model. Make sure the model has been downloaded first by pressing the 'Download Dataset' button");
            print(e);
        else:
            self.textBox.append(f"Training Done\nAccuracy: {self.accuracy :>.2f}%");