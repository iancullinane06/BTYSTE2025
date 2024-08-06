import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

class ImageViewer(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Image Channel Viewer")
        self.geometry("1200x800")
        self.create_widgets()
        self.image_files = []
        self.current_index = 0

    def create_widgets(self):
        self.image_frame = tk.Frame(self)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.label_rgb = tk.Label(self.image_frame)
        self.label_rgb.pack(side=tk.LEFT, padx=5, pady=5)

        self.channel_labels = [tk.Label(self.image_frame) for _ in range(6)]
        for label in self.channel_labels:
            label.pack(side=tk.LEFT, padx=5, pady=5)

        self.controls_frame = tk.Frame(self)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.load_button = tk.Button(self.controls_frame, text="Load Images", command=self.load_images)
        self.load_button.pack(pady=5)

        self.prev_button = tk.Button(self.controls_frame, text="Previous", command=self.show_prev_image)
        self.prev_button.pack(pady=5)

        self.next_button = tk.Button(self.controls_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(pady=5)

    def load_images(self):
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
            if self.image_files:
                self.current_index = 0
                self.show_image(self.image_files[self.current_index])

    def show_image(self, image_path):
        try:
            # Verify file existence
            if not os.path.isfile(image_path):
                print(f"File not found: {image_path}")
                return

            print(f"Loading image from path: {image_path}")  # Debugging statement
            
            # Attempt to open the image with PIL
            image = Image.open(image_path)
            image_np = np.array(image)

            if image_np.ndim == 3:
                if image_np.shape[0] == 3:
                    # Handle RGB image
                    rgb_image = np.transpose(image_np, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
                    rgb_pil = Image.fromarray(rgb_image, 'RGB')
                    rgb_imgtk = ImageTk.PhotoImage(rgb_pil)
                    self.label_rgb.config(image=rgb_imgtk)
                    self.label_rgb.image = rgb_imgtk

                    # Clear channel labels if not used
                    for label in self.channel_labels:
                        label.config(image='')

                elif image_np.shape[0] == 6:
                    # Handle 6-channel image
                    rgb_image = np.stack(image_np[:3], axis=-1)
                    rgb_pil = Image.fromarray(rgb_image, 'RGB')
                    rgb_imgtk = ImageTk.PhotoImage(rgb_pil)
                    self.label_rgb.config(image=rgb_imgtk)
                    self.label_rgb.image = rgb_imgtk

                    # Display individual channels
                    for i in range(6):
                        channel_image = Image.fromarray(image_np[i], 'L')
                        channel_imgtk = ImageTk.PhotoImage(channel_image)
                        self.channel_labels[i].config(image=channel_imgtk)
                        self.channel_labels[i].image = channel_imgtk
                else:
                    print(f"Unexpected number of channels in image at {image_path}")
                    # Clear all labels if not valid
                    self.label_rgb.config(image='')
                    for label in self.channel_labels:
                        label.config(image='')
            elif image_np.ndim == 2:
                # Handle grayscale image (single channel)
                grayscale_pil = Image.fromarray(image_np, 'L')
                grayscale_imgtk = ImageTk.PhotoImage(grayscale_pil)
                self.label_rgb.config(image=grayscale_imgtk)
                self.label_rgb.image = grayscale_imgtk

                # Clear channel labels
                for label in self.channel_labels:
                    label.config(image='')
            else:
                print(f"Image at {image_path} has an unsupported format")

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    def show_prev_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.image_files[self.current_index])

    def show_next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image(self.image_files[self.current_index])

if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()
