from age_utils import preprocess_face_image, age_from_distribution
from tensorflow.keras.models import load_model


#hada test ta3 dik i3tik lage mn chi tsswira tuploadiha
def predict_age(image_path):
    model = load_model('../Models/age_model_best.keras', compile= False)
    image = preprocess_face_image(image_path)
    age_prediction = age_from_distribution(model.predict(image, verbose=0)[0])
    print(f"Predicted Age: {int(age_prediction)}")

# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\test_data\handsome-man-looking-camera-leaning-fist.jpg')
# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\test_data\close-up-smiley-man-posing.jpg')
# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\test_data\portrait-handsome-young-man-looking-camera.jpg')
# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\test_data\portrait-handsome-bearded-man-suit.jpg')
# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\test_data\portrait-handsome-mature-man.jpg')
# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\test_data\portrait-smiley-mature-man.jpg')
# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\test_data\portrait-british-elderly-man.jpg')
# predict_age(r'C:\Users\LENOVO\Desktop\Facial-Expression-Recognition-System\Data\age_Data\UTKFace\50_1_4_20170117204033496.jpg.chip.jpg')

        
