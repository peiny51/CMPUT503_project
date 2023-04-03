import cv2  
# Import pyplot from matplotlib as plt  
from matplotlib import pyplot as plt  
# Opening the image from files  


imaging = cv2.imread("pic.png")  
img_rgb = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)
  
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()
# Use minSize because for not 
# bothering with extra-small 
# dots that would look like STOP signs
cascade_data = cv2.CascadeClassifier('crop.xml')

found = cascade_data.detectMultiScale(img_rgb, 
                                minSize =(20, 20))

# Don't do anything if there's 
# no sign
amount_found = len(found)

if amount_found != 0:
    
    # There may be more than one
    # sign in the image
    for (x, y, width, height) in found:
        
        # We draw a green rectangle around
        # every recognized sign
        cv2.rectangle(img_rgb, (x, y), 
                    (x + height, y + width), 
                    (0, 255, 0), 5)
    print(True)

else:
    print(False)
      