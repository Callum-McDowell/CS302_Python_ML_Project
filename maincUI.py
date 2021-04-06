#====== maincUI.py ======#
# The core content and 'central widget' of our app.
#


#====== Libraries ======#
from PyQt5.QtWidgets import QWidget;
from PyQt5.QtWidgets import QTextBrowser, QPushButton, QVBoxLayout;
from PyQt5.QtGui import QIcon;

import resources as r;
import peripheralUI;


#====== Main Content ======#
class AppMainContent(QWidget):
    # Our 'central widget' for the MainWindow frame.
    # Core content goes here.

    def __init__(self):
        super().__init__();
