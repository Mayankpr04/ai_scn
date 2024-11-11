import os
import sys
from PIL import Image
#check for sizes
def get_image_sizes(directory):
    sizes = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                try:
                    # Open image and get size
                    with Image.open(os.path.join(root, file)) as img:
                        width, height = img.size
                        sizes.append((file, width, height))
                        print(f"{file}: {width}x{height}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    if not sizes:
        print("No images found in the directory or unable to process images.")
    return sizes
if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_sizes.py <directory_path>")
        sys.exit(1)    
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        sys.exit(1)    
    # Get the image sizes
    image_sizes = get_image_sizes(directory_path)
    if image_sizes:
        widths = [width for _,width, _ in image_sizes]
        heights = [height for _, _, height in image_sizes]
        avg_width=sum(widths)/len(widths) if widths else 0
        avg_height=sum(heights)/len(heights) if heights else 0
        print(f"\nAverage Width: {avg_width}")
        print(f"Average Height: {avg_height}")
    else:
        print("No valid images to calculate average dimensions.")
