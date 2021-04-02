# Callum - 2nd April 2021

#====== resources.py ======#
# A header file for consolidating and simplifying access of
# static resources (e.g. icons, graphics).
# Constants are directory paths and names.

#====== Code ======#
import os;
MODULE_DIR = os.path.dirname(os.path.realpath(__file__));
RESOURCES_DIR = MODULE_DIR + "\\resources\\";
# Set up relative DIR for referencing local filepaths
# Could replace with Qt resource system, but having issues recognising pyrcc5
# https://doc.qt.io/qt-5/resources.html


#====== Icons ======#
ICON_WINDOW = RESOURCES_DIR + "icon_robot.png";
ICON_EXIT = RESOURCES_DIR + "icon_exit.svg";
ICON_OPEN = RESOURCES_DIR + "icon_open.svg";
ICON_HELP = RESOURCES_DIR + "icon_help.svg";
ICON_DRAW = RESOURCES_DIR + "icon_draw.svg";

