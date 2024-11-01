import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import pandas as pd
import os

# Initialize global variables
source_img = None
target_img = None
source_points = []
target_points = []
source_file = ""
target_file = ""
output_folder = ""
img_index = 0  # Counter for naming files

# Define a list of colors to cycle through for each pair
colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "yellow"]
color_index = 0  # Index to keep track of the current color for pairs

# Create main GUI window
root = tk.Tk()
root.title("Image Keypoint Matching Tool")

# Function to load and display images from CSV
def load_images_from_csv():
    global img_data, img_index, source_img, target_img, source_file, target_file
    csv_file = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if not csv_file:
        messagebox.showwarning("Warning", "No CSV File Selected!")
        return
    
    try:
        # Read CSV and get image paths
        img_data = pd.read_csv(csv_file)
        if "source" not in img_data.columns or "target" not in img_data.columns:
            messagebox.showerror("Error", "CSV must contain 'Source' and 'Target' columns")
            return
        
        # Reset index and load first pair
        img_index = 0
        load_next_image_pair()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load images from CSV: {e}")

# Function to load the next image pair
def load_next_image_pair():
    global source_img, target_img, source_file, target_file, img_index, color_index
    if img_index >= len(img_data):
        messagebox.showinfo("Info", "All image pairs processed.")
        return

    # Get file paths for current image pair
    source_file = img_data["source"][img_index]
    target_file = img_data["target"][img_index]
    color_index = 0  # Reset color index for each new pair

    # Display source image (resize to 600x600 and center it)
    img = Image.open(source_file)
    img = img.resize((600, 600), Image.ANTIALIAS)
    source_img = ImageTk.PhotoImage(img)
    source_canvas.delete("all")
    source_canvas.create_image(300, 300, image=source_img, anchor="center")

    # Display target image (resize to 600x600 and center it)
    img = Image.open(target_file)
    img = img.resize((600, 600), Image.ANTIALIAS)
    target_img = ImageTk.PhotoImage(img)
    target_canvas.delete("all")
    target_canvas.create_image(300, 300, image=target_img, anchor="center")

    # Clear points for the new image pair
    source_points.clear()
    target_points.clear()
    update_table()

# Function to set the output folder
def set_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        messagebox.showwarning("Warning", "No Output Folder Selected!")

# Function to handle keypoint selection on Source image
def source_click(event):
    global color_index
    x = event.x
    y = event.y
    if 0 <= x <= 600 and 0 <= y <= 600:
        source_points.append((int(x), int(y)))
        color = colors[color_index % len(colors)]
        source_canvas.create_oval(x-3, y-3, x+3, y+3, fill=color)
        if len(source_points) > len(target_points):
            messagebox.showinfo("Info", "Now select a corresponding point on the Target Image.")
        update_table()

# Function to handle keypoint selection on Target image
def target_click(event):
    global color_index
    x = event.x
    y = event.y
    if len(target_points) < len(source_points) and 0 <= x <= 600 and 0 <= y <= 600:
        target_points.append((int(x), int(y)))
        color = colors[color_index % len(colors)]
        target_canvas.create_oval(x-3, y-3, x+3, y+3, fill=color)
        color_index += 1  # Move to the next color for the next pair
        update_table()
    else:
        messagebox.showwarning("Warning", "Please select a point on the Source Image first.")

# Function to update the points table
def update_table():
    for row in points_table.get_children():
        points_table.delete(row)
    for i, (src, tgt) in enumerate(zip(source_points, target_points), start=1):
        points_table.insert("", "end", values=(i, src[0], src[1], tgt[0], tgt[1]))

# Function to save images and keypoints
def save_data():
    global img_index
    if not source_file or not target_file or not output_folder:
        messagebox.showwarning("Warning", "Ensure images and output folder are selected.")
        return
    if len(source_points) != len(target_points):
        messagebox.showwarning("Warning", "Each source point must have a corresponding target point.")
        return
    
    # Resize images to 256x256 pixels for saving
    src_img = cv2.imread(source_file)
    tgt_img = cv2.imread(target_file)
    src_img_resized = cv2.resize(src_img, (256, 256))
    tgt_img_resized = cv2.resize(tgt_img, (256, 256))

    # Scale points to fit 256x256 image size
    scaled_source_points = [(int(x * 256 / 600), int(y * 256 / 600)) for x, y in source_points]
    scaled_target_points = [(int(x * 256 / 600), int(y * 256 / 600)) for x, y in target_points]

    # Generate filenames based on img_index
    file_prefix = f"img_{img_index:03d}"
    src_save_path = os.path.join(output_folder, f"{file_prefix}_source.png")
    tgt_save_path = os.path.join(output_folder, f"{file_prefix}_target.png")
    keypoints_save_path = os.path.join(output_folder, f"{file_prefix}_keypoints.csv")

    # Save resized images
    cv2.imwrite(src_save_path, src_img_resized)
    cv2.imwrite(tgt_save_path, tgt_img_resized)

    # Save keypoints with new column names
    keypoints_data = pd.DataFrame({
        "x1": [pt[0] for pt in scaled_source_points],
        "y1": [pt[1] for pt in scaled_source_points],
        "x2": [pt[0] for pt in scaled_target_points],
        "y2": [pt[1] for pt in scaled_target_points]
    })
    keypoints_data.to_csv(keypoints_save_path, index=False)

    # Move to the next image pair
    img_index += 1
    load_next_image_pair()

# GUI layout setup
source_canvas = tk.Canvas(root, width=600, height=600, bg="white")
target_canvas = tk.Canvas(root, width=600, height=600, bg="white")
source_canvas.bind("<Button-1>", source_click)
target_canvas.bind("<Button-1>", target_click)

source_canvas.grid(row=0, column=0, padx=10, pady=10)
target_canvas.grid(row=0, column=1, padx=10, pady=10)

# Table to show selected points
points_table = ttk.Treeview(root, columns=("Point", "x1", "y1", "x2", "y2"), show="headings")
points_table.heading("Point", text="Point")
points_table.heading("x1", text="Source x")
points_table.heading("y1", text="Source y")
points_table.heading("x2", text="Target x")
points_table.heading("y2", text="Target y")
points_table.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Buttons for loading images from CSV, setting folder, saving data, and moving to the next pair
tk.Button(root, text="Load Images from CSV", command=load_images_from_csv).grid(row=2, column=0, pady=5)
tk.Button(root, text="Set Output Folder", command=set_output_folder).grid(row=2, column=1, pady=5)
tk.Button(root, text="Save & Next", command=save_data).grid(row=3, column=0, columnspan=2, pady=5)

# Run the GUI main loop
root.mainloop()
