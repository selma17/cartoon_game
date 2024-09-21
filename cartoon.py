import cv2
import numpy as np
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

# Fonction pour cartooniser une image avec des ajustements pour plus de clarté
def cartoonify_image(img):
    # Appliquer un filtre bilatéral pour réduire le bruit mais conserver les bords
    img_color = cv2.bilateralFilter(img, 7, 75, 75)  # Ajustement des paramètres pour un lissage plus faible
    
    # Conversion en niveaux de gris
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou médian
    img_blur = cv2.medianBlur(img_gray, 5)  # Augmentation du flou pour adoucir les détails
    
    # Utiliser le seuil adaptatif pour obtenir des bords
    img_edges = cv2.adaptiveThreshold(img_blur, 255, 
                                      cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 2)
    
    # Réduction du nombre de couleurs avec K-means
    data = np.float32(img_color).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 9, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    img_reduced_color = centers[labels.flatten()].reshape(img_color.shape)
    
    # Fusion des bords et de l'image réduite en couleurs
    img_cartoon = cv2.bitwise_and(img_reduced_color, img_reduced_color, mask=img_edges)
    
    # Ajuster la luminosité en augmentant la valeur des pixels
    img_cartoon = cv2.convertScaleAbs(img_cartoon, alpha=1.2, beta=30)  # Augmenter la luminosité
    
    return img_cartoon

# Capture et cartoonisation de l'image quand on appuie sur le bouton
def capture_and_cartoonize():
    global frame
    # Cartooniser l'image capturée
    cartoon_image = cartoonify_image(frame)
    
    # Sauvegarder l'image cartoonisée
    cv2.imwrite('cartoonified_output.png', cartoon_image)
    
    # Afficher l'image cartoonisée
    cartoon_image_bgr = cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cartoon_image_bgr)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    
    label_image.config(image=img_tk)
    label_image.image = img_tk  # Conserver une référence pour que l'image ne soit pas supprimée
    
    print("Image cartoonisée enregistrée sous 'cartoonified_output.png'.")

# Capture du flux vidéo
def show_video():
    global frame
    ret, frame = cap.read()
    
    if ret:
        # Convertir le frame pour l'afficher dans Tkinter
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_bgr)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Mettre à jour l'image dans l'interface
        label_image.config(image=img_tk)
        label_image.image = img_tk
    
    # Relancer la capture après un court délai pour obtenir un flux vidéo
    label_image.after(10, show_video)

# Initialiser la fenêtre Tkinter
root = tk.Tk()
root.title("Cartoonify Camera")

# Créer un bouton pour capturer et cartooniser l'image
btn = Button(root, text="Cartooniser l'image", command=capture_and_cartoonize)
btn.pack()

# Créer un label pour afficher le flux vidéo
label_image = tk.Label(root)
label_image.pack()

# Ouvrir la caméra
cap = cv2.VideoCapture(0)

# Démarrer le flux vidéo
show_video()

# Lancer l'interface Tkinter
root.mainloop()

# Libérer les ressources quand la fenêtre est fermée
cap.release()
cv2.destroyAllWindows()
