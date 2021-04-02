from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
import sys
import canvasToMNIST
import cv2
import os
  
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
  
        #Setting title
        self.setWindowTitle("Please write a number")
  
        #Setting size of the main window
        self.setGeometry(100, 100, 400, 600)
  
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

        #Clear button
        clearButton = QPushButton('Clear', self)
        clearButton.setToolTip('This is an example button')
        clearButton.move(100,500)
        clearButton.clicked.connect(self.clear)
    
        #Submit button
        submitButton = QPushButton('Submit', self)
        submitButton.setToolTip('This is an example button')
        submitButton.move(200,500)
        submitButton.clicked.connect(self.submit)
  
    #This method checks for mouse clicks
    def mousePressEvent(self, event):
  
        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
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

        #Display result here
  
#Create pyqt5 app
App = QApplication(sys.argv)
  
#Create the window instance
window = Window()
window.show()
  
sys.exit(App.exec())