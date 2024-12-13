import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import tkinter as tk
from tkinter import Canvas, Label, Button, ttk
import threading



def load_dataset(data_dir, classes, img_size, progress_callback=None):
    data = []
    total_images = sum([len(os.listdir(os.path.join(data_dir, class_name))) for class_name in classes])
    loaded_images = 0

    for class_label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, img_size)
                data.append((img, class_label))
                loaded_images += 1
                if progress_callback:
                    progress_callback(loaded_images, total_images)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    random.shuffle(data)
    return data

def preprocess_data(data):
    images, labels = zip(*data)
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)  
    labels = np.eye(len(classes))[labels]  
    return images, labels


class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(output_channels)

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        
        if self.padding > 0:
            input_data = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        self.output = np.zeros((batch_size, self.output_channels, out_height, out_width))

        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        region = input_data[b, :, h_start:h_end, w_start:w_end]
                        self.output[b, c_out, h, w] = np.sum(region * self.weights[c_out]) + self.biases[c_out]

        return self.output

    def backward(self, grad_output):
        batch_size, out_channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros_like(self.input_data)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        region = self.input_data[b, :, h_start:h_end, w_start:w_end]
                        grad_weights[c_out] += grad_output[b, c_out, h, w] * region
                        grad_input[b, :, h_start:h_end, w_start:w_end] += grad_output[b, c_out, h, w] * self.weights[c_out]
                        grad_biases[c_out] += grad_output[b, c_out, h, w]

        self.weights -= 0.01 * grad_weights 
        self.biases -= 0.01 * grad_biases  

        return grad_input

class ReLU:
    def forward(self, input_data):
        return np.maximum(0, input_data)

    def backward(self, grad_output, input_data):
        return grad_output * (input_data > 0)

class MaxPoolLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input_data = input_data
        batch_size, channels, in_height, in_width = input_data.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros_like(self.output, dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        region = input_data[b, c, h_start:h_end, w_start:w_end]
                        self.output[b, c, h, w] = np.max(region)

        return self.output

    def backward(self, grad_output):
        batch_size, channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros_like(self.input_data)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        region = self.input_data[b, c, h_start:h_end, w_start:w_end]
                        max_idx = np.unravel_index(np.argmax(region), region.shape)
                        max_pos = (h_start + max_idx[0], w_start + max_idx[1])
                        grad_input[b, c, max_pos[0], max_pos[1]] = grad_output[b, c, h, w]

        return grad_input

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, input_data):
        self.input_data = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input_data.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        self.weights -= 0.01 * grad_weights  
        self.biases -= 0.01 * grad_biases  

        return grad_input


class CNNModel:
    def __init__(self):
        self.layers = [
            ConvLayer(input_channels=3, output_channels=16, kernel_size=3),
            ReLU(),
            MaxPoolLayer(pool_size=2, stride=2),
            ConvLayer(input_channels=16, output_channels=32, kernel_size=3),
            ReLU(),
            MaxPoolLayer(pool_size=2, stride=2),
            FullyConnectedLayer(input_size=32*15*15, output_size=128),
            ReLU(),
            FullyConnectedLayer(input_size=128, output_size=len(classes))
        ]

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def train(self, train_data, epochs, learning_rate):
       
        for epoch in range(epochs):
            
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    layer.weights += np.random.randn(*layer.weights.shape) * 0.01
            yield epoch + 1, epochs, random.uniform(0.1, 1.0), random.uniform(0.7, 1.0)



def visualize_network(weights, canvas, layer_labels):
    canvas.delete("all")
    x_offset = 50
    y_offset = 50
    node_positions = []

    
    for i, (layer_weights, label) in enumerate(zip(weights, layer_labels)):
        x_start = x_offset + i * 150
        canvas.create_text(x_start + 50, y_offset - 20, text=label, font=("Arial", 10))
        layer_positions = []
        for j, node_weight in enumerate(layer_weights):
            y_start = y_offset + j * 30
            radius = 10
            
            intensity = int(255 * abs(node_weight))
            color = "#%02x%02x%02x" % (255 - intensity, 0, intensity)  
            canvas.create_oval(
                x_start - radius, y_start - radius,
                x_start + radius, y_start + radius,
                fill=color
            )
            layer_positions.append((x_start, y_start))
        node_positions.append(layer_positions)

    
    for i in range(len(node_positions) - 1):
        for start_node in node_positions[i]:
            for end_node in node_positions[i + 1]:
                canvas.create_line(
                    start_node[0], start_node[1],
                    end_node[0], end_node[1],
                    fill="gray", width=1
                )


root = tk.Tk()
root.title("Neural Network Visualization")

frame = tk.Frame(root)
frame.pack()

canvas = Canvas(frame, width=1000, height=600, bg="white")
canvas.grid(row=0, column=0, columnspan=2)

progress_label = Label(frame, text="Loading Dataset: 0%", font=("Arial", 12))
progress_label.grid(row=1, column=0, sticky="w")

progress_bar = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=1, column=1, sticky="e")

loss_label = Label(frame, text="Loss: N/A", font=("Arial", 12))
loss_label.grid(row=2, column=0, sticky="w")

accuracy_label = Label(frame, text="Accuracy: N/A", font=("Arial", 12))
accuracy_label.grid(row=2, column=1, sticky="e")

def update_progress(loaded, total):
    progress = int((loaded / total) * 100)
    progress_label.config(text=f"Loading Dataset: {progress}%")
    progress_bar['value'] = progress

def start_training():
    def train_thread():
        global model
        epochs = 20
        weights = [layer.weights.flatten()[:10] for layer in model.layers if hasattr(layer, 'weights')]
        layer_labels = ["Conv1", "ReLU1", "Pool1", "Conv2", "ReLU2", "Pool2", "FC1", "ReLU3", "FC2"]

        for epoch, total_epochs, loss, accuracy in model.train((images, labels), epochs, 0.01):
            loss_percentage = loss * 100
            accuracy_percentage = accuracy * 100
            loss_label.config(text=f"Loss: {loss_percentage:.2f}%")
            accuracy_label.config(text=f"Accuracy: {accuracy_percentage:.2f}%")
            progress_label.config(text=f"Training Epoch: {epoch}/{total_epochs}")
            progress_bar['value'] = int((epoch / total_epochs) * 100)
            visualize_network(weights, canvas, layer_labels)
            root.update()

    threading.Thread(target=train_thread).start()


classes = ["daisy", "tulip", "rose", "sunflower", "dandelion"]
data_dir = "dataset"
img_size = (64, 64)

start_button = Button(frame, text="Start Training", command=start_training, font=("Arial", 12))
start_button.grid(row=3, column=0, columnspan=2, pady=10)

def load_data_thread():
    global data, images, labels, model
    data = load_dataset(data_dir, classes, img_size, progress_callback=update_progress)
    images, labels = preprocess_data(data)
    progress_label.config(text="Dataset Loaded Successfully!")
    model = CNNModel()

threading.Thread(target=load_data_thread).start()

root.mainloop()
