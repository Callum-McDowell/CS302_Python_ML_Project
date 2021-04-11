#====== maincUI.py ======#
# The core content and 'central widget' of our app.
#


#====== Libraries ======#
from PyQt5.QtWidgets import *;
from PyQt5.QtCore import * 
from PyQt5.QtGui import *;

import resources as r;
import peripheralUI;
import sys
import canvasToMNIST
import prediction
import cv2
import os
import os.path
from os import path
from PIL import Image
import matplotlib.pyplot as plt

class Canvas(QWidget):
    def __init__(self):
        super().__init__();

     #Setting title
        self.setWindowTitle("Please write a number")

        self.setFixedSize(400, 400)
  
        #Creating image object
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        #drawing flag
        self.drawing = False
        #brush size
        self.brushSize = 10
        #color
        self.brushColor = Qt.black
  
        #QPoint object to track the point
        self.lastPoint = QPoint()

    #This method checks for mouse clicks
    def mousePressEvent(self, event):
  
        #Check if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            #Make drawing flag true
            self.drawing = True
            #Make the last point at the mouse cursor position
            self.lastPoint = event.pos()
  
    #This method tracks mouse activity
    def mouseMoveEvent(self, event):
          
        #Checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
              
            #Creating painter object
            painter = QPainter(self.image)
              
            #Set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize, 
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
              
            #Draw line from the last point of cursor to the current point
            painter.drawLine(self.lastPoint, event.pos())
              
            #Change the last point
            self.lastPoint = event.pos()
            #Update
            self.update()
  
    #Method called when mouse button is released
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
  
    #Paint event
    def paintEvent(self, event):
        #Create a canvas
        canvasPainter = QPainter(self)
          
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
  
  
    #Method for clearing everything on canvas
    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

    def submit(self):
        #Saving QImage to a file first as opencv cannot process QImage
        self.image.save("input.png")

        #Reading image with opencv
        img = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

        #Removing file to save storage space
        os.remove("input.png")

        #Converting image to MNIST format
        img = canvasToMNIST.cropInput(img)
        img = canvasToMNIST.convertToMNIST(img)
        return img

#====== Main Content ======#
class AppMainContent(QWidget):
    # Our 'central widget' for the MainWindow frame.
    # Core content goes here.
    def __init__(self, model):
        super().__init__();
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.canvas = Canvas()
        self.layout.addWidget(self.canvas)
        self.textBox = QTextEdit(self)
        self.textBox.setReadOnly(True)
        self.model = model
        
        #Change model button
        changeModelButton = QPushButton('Change model', self)
        changeModelButton.move(450, 50)
        changeModelButton.clicked.connect(self.changeModel)

        #Clear button
        clearButton = QPushButton('Clear', self)
        clearButton.move(450, 100)
        clearButton.clicked.connect(self.clear)
    
        #Submit button
        submitButton = QPushButton('Submit', self)
        submitButton.move(450, 150)
        submitButton.clicked.connect(self.submit)

        #Displaying prediction and probability graph
        self.textBox.move(450, 200)
        self.showGraphButton =  QPushButton('Show graph', self)
        self.showGraphButton.move(450, 250)
        self.showGraphButton.hide()
        self.showGraphButton.clicked.connect(self.showPlot)

    def submit(self):
        #Exception would be executed if no input is found
        try:
            img = self.canvas.submit()
        except:
            return

        #Displaying result
        pred, self.plt = prediction.predict(img, self.model)
        self.textBox.setText(str(pred))
        self.showGraphButton.show()

    def clear(self):
        #Close plot if it's still open
        try:
            self.plt.close()
        except Exception:
            pass
        finally:
            self.textBox.setText("")
            self.canvas.clear()

    #Show probability graph when the "show graph" button is clicked
    def showPlot(self):
        mngr = self.plt.get_current_fig_manager()
        mngr.window.setGeometry(50,100,640, 545)
        self.plt.show()

    #Change current model
    def changeModel(self):
        modelFilename, _ = QFileDialog.getOpenFileName(self,"Please select model", "Model/","pickle files (*.pkl)")
        if len(modelFilename) > 0:
            self.model = modelFilename

  
        

