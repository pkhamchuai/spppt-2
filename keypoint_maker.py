import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import os

# Global variables
source_img = None
target_img = None
source_points = []
target_points = []
output_folder = ""
img_index = 0
img_data = None
output_log = "image_pair_log.csv"
main_log_df = pd.DataFrame(columns=["source", "target", "keypoints", "training"])

# Global variable to store the selected keypoint index
selected_point_index = None

# Define colors for pairs
colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "yellow"]
color_index = 0

# Initialize GUI
root = tk.Tk()
root.title("Image Keypoint Matching Tool")

canvas_size = 600 # for 1080p screen
# canvas_size = 400 # for 900p screen
canvas_center = canvas_size // 2

# Functions to set output folder and CSV file
def set_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        messagebox.showwarning("Warning", "No Output Folder Selected!")
    else:
        output_folder_label.config(text=f"Output Folder: {output_folder}")

def load_images_from_csv():
    global img_data, img_index, output_folder
    csv_file = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    if not csv_file:
        messagebox.showwarning("Warning", "No CSV File Selected!")
        return

    try:
        img_data = pd.read_csv(csv_file)
        img_data = img_data[img_data['training'] == 0]
        if "source" not in img_data.columns or "target" not in img_data.columns:
            messagebox.showerror("Error", "CSV must contain 'Source' and 'Target' columns")
            return
        if not output_folder:
            messagebox.showwarning("Warning", "Set an output folder first.")
            return
        img_index = 0
        load_image_pair()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load images from CSV: {e}")

def load_image_pair():
    global source_img, target_img, img_index, color_index
    global source_file, target_file
    if img_index < 0 or img_index >= len(img_data):
        messagebox.showinfo("Info", "Image pair out of range.")
        return

    source_file = img_data["source"].iloc[img_index]
    target_file = img_data["target"].iloc[img_index]
    color_index = 0

    source_img_path = os.path.join(output_folder, f"img_{img_index:03d}_source.png")
    target_img_path = os.path.join(output_folder, f"img_{img_index:03d}_target.png")

    def load_image(img_path):
        img = Image.open(img_path)
        img = img.resize((canvas_size, canvas_size), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    source_img = load_image(source_file)
    target_img = load_image(target_file)
    source_canvas.create_image(canvas_center, canvas_center, image=source_img, anchor="center")
    target_canvas.create_image(canvas_center, canvas_center, image=target_img, anchor="center")
    keypoints_file = os.path.join(output_folder, f"img_{img_index:03d}_keypoints.csv")

    # Check for keypoints file existence, fallback to img_data["keypoints"]
    if os.path.exists(keypoints_file):
        load_keypoints_from_file(keypoints_file)
        display_existing_points()  # Show the loaded points on the canvases
    elif "keypoints" in img_data.columns and pd.notna(img_data["keypoints"].iloc[img_index]):
        alt_keypoints_file = img_data["keypoints"].iloc[img_index]
        if os.path.exists(alt_keypoints_file):
            load_keypoints_from_file(alt_keypoints_file)
            display_existing_points()  # Show the loaded points on the canvases
        else:
            messagebox.showinfo("Info", f"Keypoints file '{alt_keypoints_file}' does not exist.")
            source_points.clear()
            target_points.clear()
    else:
        source_points.clear()
        target_points.clear()

    update_table()
    image_pair_label.config(text=f"Image Pair {img_index + 1} / {len(img_data)}")

def trim_to_last_two_levels(path):
    # keep only the last two levels of the path
    path = path.split('/')
    path = path[-3:]
    path = '/'.join(path)
    return path
    
def load_keypoints_from_file(keypoints_file):
    global source_points, target_points
    points_df = pd.read_csv(keypoints_file)
    source_points = [(int(row["x1"] * canvas_size / 256), int(row["y1"] * canvas_size / 256)) for idx, row in points_df.iterrows()]
    target_points = [(int(row["x2"] * canvas_size / 256), int(row["y2"] * canvas_size / 256)) for idx, row in points_df.iterrows()]

def save_keypoints():
    global img_index
    keypoints_file = os.path.join(output_folder, f"img_{img_index:03d}_keypoints.csv")
    keypoints_file = trim_to_last_two_levels(keypoints_file)

    points_df = pd.DataFrame({
        "x1": [p[0] * 256 / canvas_size for p in source_points],
        "y1": [p[1] * 256 / canvas_size for p in source_points],
        "x2": [p[0] * 256 / canvas_size for p in target_points],
        "y2": [p[1] * 256 / canvas_size for p in target_points]
    })
    points_df.to_csv(keypoints_file, index=False)
    main_log_df.loc[img_index] = [source_file, target_file, keypoints_file, 0]
    main_log_df.to_csv(os.path.join(output_folder, output_log), index=False)

def next_image_pair():
    global img_index
    save_keypoints()
    if img_index < len(img_data) - 1:
        img_index += 1
        load_image_pair()
    else:
        messagebox.showinfo("Info", "This is the last image pair.")

def prev_image_pair():
    global img_index
    save_keypoints()
    if img_index > 0:
        img_index -= 1
        load_image_pair()
    else:
        messagebox.showinfo("Info", "This is the first image pair.")

def source_click(event):
    global color_index
    x, y = event.x, event.y
    source_points.append((x, y))
    source_canvas.create_oval(x-3, y-3, x+3, y+3, fill=colors[color_index % len(colors)], tags="point")
    update_table()

def target_click(event):
    global color_index
    x, y = event.x, event.y
    target_points.append((x, y))
    target_canvas.create_oval(x-3, y-3, x+3, y+3, fill=colors[color_index % len(colors)], tags="point")
    color_index += 1
    update_table()

# Function to highlight selected keypoint pair
def on_select(event):
    global selected_point_index
    selected_item = points_table.selection()
    if selected_item:
        selected_point_index = int(points_table.item(selected_item)["values"][0]) - 1
        display_existing_points()  # Refresh display to show selection

def remove_selected_pair():
    global selected_point_index
    selected_item = points_table.selection()
    if not selected_item:
        messagebox.showwarning("Warning", "No keypoint pair selected.")
        return

    index = int(points_table.item(selected_item)["values"][0]) - 1
    if 0 <= index < len(source_points) and 0 <= index < len(target_points):
        del source_points[index]
        del target_points[index]
        selected_point_index = None
        update_table()
        display_existing_points()
    else:
        messagebox.showerror("Error", "Selected index out of range.")

def update_table():
    for item in points_table.get_children():
        points_table.delete(item)
    
    for i, (sp, tp) in enumerate(zip(source_points, target_points)):
        points_table.insert("", "end", values=(i + 1, sp[0], sp[1], tp[0], tp[1]))

def display_existing_points():
    """Display points on both source and target canvases with a white stroke for the selected pair."""
    source_canvas.delete("point")
    target_canvas.delete("point")
    for i, (sp, tp) in enumerate(zip(source_points, target_points)):
        color = colors[i % len(colors)]
        # Draw the main dot
        source_canvas.create_oval(sp[0] - 3, sp[1] - 3, sp[0] + 3, sp[1] + 3, fill=color, tags="point")
        target_canvas.create_oval(tp[0] - 3, tp[1] - 3, tp[0] + 3, tp[1] + 3, fill=color, tags="point")
        
        # Draw white stroke if this pair is selected
        if i == selected_point_index:
            source_canvas.create_oval(sp[0] - 5, sp[1] - 5, sp[0] + 5, sp[1] + 5, outline="white", width=2, tags="point")
            target_canvas.create_oval(tp[0] - 5, tp[1] - 5, tp[0] + 5, tp[1] + 5, outline="white", width=2, tags="point")

# GUI layout
source_canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
target_canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
source_canvas.bind("<Button-1>", source_click)
target_canvas.bind("<Button-1>", target_click)
source_canvas.grid(row=0, column=0, padx=10, pady=10)
target_canvas.grid(row=0, column=1, padx=10, pady=10)

points_table = ttk.Treeview(root, columns=("Point", "x1", "y1", "x2", "y2"), show="headings")
points_table.heading("Point", text="Point")
points_table.heading("x1", text="Source x")
points_table.heading("y1", text="Source y")
points_table.heading("x2", text="Target x")
points_table.heading("y2", text="Target y")
points_table.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
# points_table.bind("<ButtonRelease-1>", on_select)
points_table.column("Point", width=50, anchor="center")
points_table.column("x1", width=100, anchor="center")
points_table.column("y1", width=100, anchor="center")
points_table.column("x2", width=100, anchor="center")
points_table.column("y2", width=100, anchor="center")
points_table.tag_configure("selected", background="lightblue")

# Bind selection event to the table
points_table.bind("<<TreeviewSelect>>", on_select)

image_pair_label = tk.Label(root, text="Image Pair 0 / 0")
image_pair_label.grid(row=2, column=0, pady=5)

tk.Button(root, text="Load Images from CSV", command=load_images_from_csv).grid(row=3, column=0, pady=5)
tk.Button(root, text="Set Output Folder", command=set_output_folder).grid(row=3, column=1, pady=5)
tk.Button(root, text="Previous Pair", command=prev_image_pair).grid(row=4, column=0, pady=5)
tk.Button(root, text="Next Pair", command=next_image_pair).grid(row=4, column=1, pady=5)
output_folder_label = tk.Label(root, text="Output Folder: Not Set")
output_folder_label.grid(row=5, column=0, columnspan=2)

# Add to the GUI layout
remove_button = tk.Button(root, text="Remove Selected Pair", command=remove_selected_pair)
remove_button.grid(row=5, column=1, pady=5)

root.mainloop()
