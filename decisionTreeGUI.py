import tkinter as tk
from tkinter import filedialog
import pandas as pd
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
import decisionTree


class DecisionTreeGUI:

    def __init__(self):
        self.dt = None
        self.window = tk.Tk()
        self.window.title("Decision Tree GUI")

        self.csv_file_label = tk.Label(self.window, text="CSV File:")
        self.csv_file_label.pack()
        self.csv_file_entry = tk.Entry(self.window)
        self.csv_file_entry.pack()

        self.browse_button = tk.Button(self.window, text="Browse", command=self.browse_csv_file)
        self.browse_button.pack()

        self.train_button = tk.Button(self.window, text="Train", command=self.train)
        self.train_button.pack()

        self.data_to_predict_label = tk.Label(self.window, text="Data to Predict:")
        self.data_to_predict_label.pack()
        self.data_to_predict_entry = tk.Entry(self.window)
        self.data_to_predict_entry.pack()

        self.predict_button = tk.Button(self.window, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.generate_pdf_button = tk.Button(self.window, text="Generate PDF", command=self.generate_pdf)
        self.generate_pdf_button.pack()

        self.window.mainloop()

    def browse_csv_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.csv_file_entry.delete(0, tk.END)
            self.csv_file_entry.insert(tk.END, file_path)

    def train(self):
        csv_file_path = self.csv_file_entry.get()

        if csv_file_path:
            dataset = pd.read_csv(csv_file_path)
            print(dataset)
            self.dt = decisionTree.DecisionTree(dataset)
            self.dt.train()

            print("Training completed.")
        else:
            print("Please select a CSV file.")

    def predict(self):
        data_to_predict = self.data_to_predict_entry.get()
        if data_to_predict:
            data_to_predict = eval(data_to_predict)  # Convert string to list
            predicted_output = self.dt.predict(data_to_predict)
            print("Predicted output: ", predicted_output)
        else:
            print("Please enter the data to predict.")

    def generate_pdf(self):
        if hasattr(self, 'dt'):
            tree.plot_tree(self.dt.dtc)
            plt.show()
            dot_data = tree.export_graphviz(self.dt.dtc, out_file=None,
                                            feature_names=self.dt.df_encoded.columns[:-1],
                                            class_names=self.dt.label_encoder.classes_,
                                            filled=True, rounded=True)
            graph = graphviz.Source(dot_data)
            graph.render("mytree1")
            print("PDF generated.")
        else:
            print("Please train the model first.")
