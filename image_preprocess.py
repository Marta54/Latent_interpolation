# image preprocess
# the images that the GAN receives is a square

from PIL import Image

def make_square(image_path, size):
    # Open the image
    img = Image.open(image_path)
    
    # Calculate the padding required to make the image square
    width, height = img.size
    if width == height:
        # Resize the image directly if it's already square
        img = img.resize((size, size), Image.BICUBIC)
    else:
        # Pad the image to make it square
        img = img.resize((size, size), Image.BICUBIC)
    
    return img

# Usage Example:
# squared_image = make_image_square("path_to_your_image.jpg", 500)
# squared_image.show()  # To display the image
# squared_image.save("squared_image.jpg")  # To save the image
