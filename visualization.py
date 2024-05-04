import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime


def start_time():
    project_start_time = datetime.now()
    return project_start_time


def end_time():
    project_end_time = datetime.now()
    return project_end_time


def calculate_time(project_start_time, project_end_time):
    print("Start time:", project_start_time)
    print("end time:", project_end_time)
    duration = project_end_time - project_start_time
    duration_in_seconds = duration.total_seconds()
    print("Duration in seconds:", duration_in_seconds)
    duration_in_minutes = duration_in_seconds / 60
    print("Duration in minutes:", duration_in_minutes)


def save_variables_to_file(dataset_folder, annotations_path, start_time_in_func, end_time_in_func, loss,
                           val_loss, train_accuracy, val_accuracy, precision, recall, f1):
    save_dir = r"C:\Users\19116\Desktop"  # Path to file

    # Check if the save directory exists, create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the next available save index
    save_index = 1
    while os.path.exists(os.path.join(save_dir, f"save{save_index}.txt")):
        save_index += 1

    # Construct the file path
    file_path = os.path.join(save_dir, f"save{save_index}.txt")

    # Write variables to the file
    with open(file_path, 'w') as file:
        file.write(f"Program Start Time: {start_time_in_func}\n")
        file.write(f"Program End Time: {end_time_in_func}\n")
        file.write(f"Dataset Folder: {dataset_folder}\n")
        file.write(f"Annotations Path: {annotations_path}\n")
        file.write(f"Loss: {loss}\n")
        file.write(f"Valid Loss: {val_loss}\n")
        file.write(f"Train Accuracy: {train_accuracy}\n")
        file.write(f"Valid Accuracy: {val_accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")


# Through history to calculate metrics
def calculate_metrics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    precision = history.history['precision']
    recall = history.history['recall']
    f1 = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]

    return loss, val_loss, train_accuracy, val_accuracy, precision, recall, f1


def print_all_metrics(loss, val_loss, train_accuracy, val_accuracy, precision, recall, f1):
    print("All Epochs Training Loss:")
    print(["{:.2f}%".format(loss_item * 100) for loss_item in loss])
    print("All Epochs Validation Loss:")
    print(["{:.2f}%".format(val_loss_item * 100) for val_loss_item in val_loss])
    print("All Epochs Training Accuracy:")
    print(["{:.2f}%".format(acc_item * 100) for acc_item in train_accuracy])
    print("All Epochs Validation Accuracy:")
    print(["{:.2f}%".format(val_acc_item * 100) for val_acc_item in val_accuracy])
    print("All Epochs Precision:")
    print(["{:.2f}%".format(precision_item * 100) for precision_item in precision])
    print("All Epochs Recall:")
    print(["{:.2f}%".format(recall_item * 100) for recall_item in recall])
    print("All Epochs F1 Score:")
    print(["{:.2f}%".format(f1_item * 100) for f1_item in f1])

    print("Final loss: {:.2f}%".format(loss[-1] * 100))
    print("Final val_loss: {:.2f}%".format(val_loss[-1] * 100))
    print("Final train_accuracy: {:.2f}%".format(train_accuracy[-1] * 100))
    print("Final val_accuracy: {:.2f}%".format(val_accuracy[-1] * 100))
    print("Final precision: {:.2f}%".format(precision[-1] * 100))
    print("Final recall: {:.2f}%".format(recall[-1] * 100))
    print("Final f1: ", f1[-1])


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(25, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# Plot training history
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()


def plot_metrics_history(train_accuracy, val_accuracy, precision, recall, f1):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.plot(train_accuracy, color='blue')
    plt.plot(val_accuracy, color='orange')
    plt.plot(precision, color='green')
    plt.plot(recall, color='red')
    plt.plot(f1, color='purple')
    plt.title('Model metrics')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Train Accuracy', 'Val Accuracy', 'Precision', 'Recall', 'F1'], loc='upper left')

    plt.show()


def plot_show_sample(y_true_classes, y_pred_classes, x_test, class_names, num_samples):

    indices = np.random.choice(len(y_true_classes), num_samples, replace=False)
    sample_true_classes = y_true_classes[indices]
    sample_pred_classes = y_pred_classes[indices]
    sample_images = x_test[indices]

    sample_images = (
            (sample_images - sample_images.min()) / (sample_images.max() - sample_images.min()) * 255).astype(
        np.uint8)

    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplots_adjust(wspace=0.4)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_images[i])
        plt.title(f"True: {class_names[sample_true_classes[i]]}\nPre: {class_names[sample_pred_classes[i]]}")
        plt.axis('off')

    plt.show()


# print visualization
def visualization(history, class_names, dataset_folder, annotations_path, start_time_in_func,
                  end_time_in_func, y_true_classes, y_pred_classes, x_test, num_samples=5):

    loss, val_loss, train_accuracy, val_accuracy, precision, recall, f1 = calculate_metrics(history)

    # saving function
    save_variables_to_file(dataset_folder, annotations_path, start_time_in_func, end_time_in_func, loss,
                           val_loss, train_accuracy, val_accuracy, precision, recall, f1)

    # print function
    print_all_metrics(loss, val_loss, train_accuracy, val_accuracy, precision, recall, f1)

    # plot function
    plot_metrics_history(train_accuracy, val_accuracy, precision, recall, f1)
    plot_training_history(history)
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names)

    # Select a subset of samples for visualization
    plot_show_sample(y_true_classes, y_pred_classes, x_test, class_names, num_samples)
