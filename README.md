
# Overview
This repository contains two Python scripts implementing a decision tree classifier and a graphical user interface (GUI) for it.     
The decision tree script (decisionTree.py) provides a custom class for training and predicting with a decision tree model using Scikit-Learn.    


The GUI script (decisionTreeGUI.py) offers a user-friendly interface to interact with the decision tree, including functions to load data, train the model, predict outcomes, and visualize the decision tree.

# Files
### decisionTree.py:
 Implements the DecisionTree class, which includes methods for data preprocessing, entropy calculation, feature splitting, model training, and prediction.

### decisionTreeGUI.py:
 Builds a GUI using Tkinter, integrating the DecisionTree class for data loading, model training, prediction, and tree visualization.

## Dependencies  
sklearn  
numpy  
pandas  
tkinter  
graphviz  
matplotlib  

## Usage
#### Training and Predicting with Decision Tree:  
Import DecisionTree from decisionTree.py.  
Create an instance with a dataset.  
Call train method to train the model.  
Use predict method for predictions.  

#### Using the GUI:
Run decisionTreeGUI.py script.  
Use the GUI to load a CSV file, train the model, make predictions, and generate a PDF of the decision tree.
#### Features  
Custom Decision Tree Implementation: Allows entropy-based feature splitting and training with Scikit-Learn's DecisionTreeClassifier.  
GUI for User Interaction: Facilitates non-programmers to use the decision tree model for predictions and visualization.  
