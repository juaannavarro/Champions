from PIL import Image
import numpy as np

# Load the image
img = Image.open("/Users/juanlu_navarro/Documents/Carrera Juan/programacion/Champions/flag.jpg")

# Convert the image to numpy array
img_array = np.array(img)

# Function to hash the images for comparison since direct pixel comparison
# might be too strict due to potential compression artifacts.
def hash_image(image):
    # Resize the image to 8x8 to reduce detail
    image = image.resize((8, 8), Image.ANTIALIAS)
    # Convert to grayscale
    image = image.convert("L")
    # Create the hash array
    pixels = np.array(image)
    # Flatten the array
    flat_pixels = pixels.flatten()
    # Calculate the median
    median = np.median(flat_pixels)
    # Create the hash based on whether each pixel is above or below the median
    return ''.join(['1' if pixel > median else '0' for pixel in flat_pixels])

# Divide the collage into individual images assuming they are all of equal size
# and in a grid layout
def divide_into_images(array, rows, cols):
    # Calculate the height and width of each subimage
    subimage_height = array.shape[0] // rows
    subimage_width = array.shape[1] // cols
    
    # Initialize an empty list to store subimages
    subimages = []
    
    # Cut the image into subimages
    for i in range(0, rows):
        for j in range(0, cols):
            # Calculate the boundaries of the subimage
            top = i * subimage_height
            left = j * subimage_width
            bottom = (i + 1) * subimage_height
            right = (j + 1) * subimage_width
            
            # Extract the subimage
            subimage = array[top:bottom, left:right]
            # Convert to a PIL image
            subimage_pil = Image.fromarray(subimage)
            # Append to the list with a hash
            subimages.append((subimage_pil, hash_image(subimage_pil)))
    
    return subimages

# Assuming a grid layout, let's try to guess the layout by looking at the aspect ratio
aspect_ratio = img_array.shape[1] / img_array.shape[0]
# Assuming the layout is more wide than tall
cols_guess = round(aspect_ratio * 10)
rows_guess = 10

# Divide the image and get the list of subimages with their hashes
subimages_with_hashes = divide_into_images(img_array, rows_guess, cols_guess)

# Count unique images by their hash
unique_images_hashes = set(hash for _, hash in subimages_with_hashes)

# Return the count of unique images and the hashes for further analysis
len(unique_images_hashes), unique_images_hashes
# Now, let's identify the image corresponding to the value "6" which is the top-left corner image
# and compare other images to find repetitions or patterns.

# Extract the top-left corner image hash
hash_image_6 = hash_image(subimages_with_hashes[0][0])

# We will create a dictionary to store images hashed as keys and the number of occurrences as values
image_occurrences = {}

# Populate the dictionary with occurrences
for _, hash_code in subimages_with_hashes:
    if hash_code in image_occurrences:
        image_occurrences[hash_code] += 1
    else:
        image_occurrences[hash_code] = 1

# Now, let's find the image that does not have a hexadecimal representation by excluding
# the ones that repeat in a pattern consistent with hexadecimal (0-F, 16 images, assuming one image
# does not belong to hexadecimal values and given the clue that one image is '6', we expect some
# images to repeat exactly or close to the number of times that would fit into the grid minus one for the non-hex image).
# We expect the non-repeating image to have a distinctly lower count of occurrences.

# First, let's see how many times the '6' image occurs to get a baseline for our pattern
occurrences_of_6 = image_occurrences[hash_image_6]

# Find the image(s) that do not fit the pattern
non_hex_image_hash = None
for image_hash, occurrences in image_occurrences.items():
    if abs(occurrences - occurrences_of_6) > 1:  # Allowing some margin for error
        non_hex_image_hash = image_hash
        break

non_hex_image_hash, occurrences_of_6, image_occurrences[non_hex_image_hash] if non_hex_image_hash else 'No distinct non-hex image found'
