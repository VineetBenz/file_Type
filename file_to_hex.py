import os
import csv
import binascii
image_dir = "png"
text_dir = "txt"

def file_to_hex(filepath):
    with open(filepath, 'rb') as file:
        content = file.read()
        hex_data = binascii.hexlify(content).decode('utf-8')
    return hex_data

# Prepare CSV data
data = []

# Process image files and label as 0
for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):  # Add more extensions if needed
        filepath = os.path.join(image_dir, filename)
        hex_data = file_to_hex(filepath)  # Now the function should be recognized
        data.append([hex_data, 0])  # Label 0 for images

# Process text files and label as 1
for filename in os.listdir(text_dir):
    if filename.endswith('.txt'):  # Add more text file extensions if needed
        filepath = os.path.join(text_dir, filename)
        hex_data = file_to_hex(filepath)  # Now the function should be recognized
        data.append([hex_data, 1])  # Label 1 for text files

# Save to CSV
csv_filename = 'file_data.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Hex Code", "Label"])  # Write header
    writer.writerows(data)

print(f"Data saved to {csv_filename}")