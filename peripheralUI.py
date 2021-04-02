# Callum - 2nd April 2021

#====== peripheralUI.py ======#
# Common UI features that may be used in other modules,
# but are not standalone. Typically be populated with
# content when initialised. e.g. popup info boxes.

#====== Libraries ======#
from PyQt5.QtWidgets import QWidget;
from PyQt5.QtWidgets import QTextBrowser, QPushButton, QVBoxLayout;
from PyQt5.QtGui import QIcon;

import resources as r;


#====== Code ======#

# PopupBox()
# Basic low priority box that displays rich-format text in a box,
# and can be closed when the user chooses. Independent window.
class PopupBox(QWidget):
    # Text box that pops up in a new windows.
    # Useful for displaying reports and detailed information.
    
    def __init__(self):
        super().__init__();
        self.initPopup();

    def initPopup(self):
        self.setWindowTitle("Popup Box");
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