import numpy as np
from sklearn.model_selection import train_test_split

from dataset import load_dataset, processing_2_interpolation
from Train_MobileNet import build_model, train_model_with_generator
from visualization import start_time, end_time, calculate_time, visualization


start_time = start_time()

# Path to file
# dataset_folder = r"G:\大学\大四\Project\archive\images"
# annotations_path = r"G:\大学\大四\Project\archive\annotations.csv"
dataset_folder = r'G:\大学\大四\毕业论文\数据集\数据扩充数据\images'
annotations_path = r'G:\大学\大四\毕业论文\数据集\数据扩充数据\augmented_annotations.csv'

# Load dataset and csv file
image_paths, labels, labels_categorical, class_names = load_dataset(dataset_folder, annotations_path)

# Image processing function
processed_images = np.array(list(processing_2_interpolation(image_paths)))

# Divide function
X_train, X_test, y_train, Y_test = train_test_split(processed_images, labels_categorical, test_size=0.2,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Build model
num_classes = len(np.unique(labels))
model = build_model(num_classes)
model.summary()

# Train model
history = train_model_with_generator(model, X_train, y_train, X_val, y_val)

# Predictions for the test set
y_pred_probs = model.predict(X_test)
y_true_classes = np.argmax(Y_test, axis=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

end_time = end_time()
calculate_time(start_time, end_time)

# Visualization
visualization(history, class_names, dataset_folder, annotations_path, start_time, end_time, y_true_classes,
              y_pred_classes, X_test)
