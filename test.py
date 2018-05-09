from keras.models import load_model
import cv2
import numpy as np
import os

train_data_dir = "images-train"
val_data_dir = "images-val-pub"

classes = {'daniel_craig': 31, 'cary_grant': 25, 'nikola_tesla': 79, 
		   'oprah_winfrey': 81, 'adam_levine': 0, 'beyonce': 15, 
		   'madeleine_albright': 66, 'marie_curie': 68, 'gwyneth_paltrow': 45, 
		   'bruce_springsteen': 21, 'mila_kunis': 74, 'ben_affleck': 13, 
		   'bill_clinton': 16, 'donald_trump': 34, 'amy_poehler': 4, 
		   'theodore_roosevelt': 93, 'albert_einstein': 3, 'hillary_clinton': 48, 
		   'natalie_portman': 77, 'al_gore': 2, 'j.k._simmons': 50, 
		   'billy_joel': 17, 'angela_merkel': 6, 'humphrey_bogart': 49, 
		   'steven_spielberg': 91, 'miley_cyrus': 75, 'chrissy_teigen': 29, 
		   'sheryl_sandberg': 90, 'aretha_franklin': 8, 'al_franken': 1, 
		   'cristiano_ronaldo': 30, 'mark_wahlberg': 69, 'noam_chomsky': 80, 
		   'jennifer_aniston': 51, 'vladimir_putin': 96, 'mikhail_gorbachev': 73, 
		   'hank_williams': 47, 'angelina_jolie': 7, 'arnold_schwarzenegger': 10, 
		   'bob_dylan': 18, 'ellen_page': 36, 'halle_berry': 46, 
		   'carly_fiorina': 24, 'john_mccain': 56, 'barack_obama': 12, 
		   'orlando_bloom': 82, 'johnny_depp': 58, 'lady_gaga': 63, 
		   'johnny_cash': 57, 'michelle_obama': 72, 'kanye_west': 60, 
		   'garth_brooks': 42, 'franklin_roosevelt': 41, 'mitt_romney': 76, 
		   'ryan_gosling': 86, 'demi_moore': 33, 'ronald_reagan': 85, 
		   'celine_dion': 26, 'selena_gomez': 89, 'tom_hanks': 95, 
		   'kim_kardashian': 62, 'britney_spears': 20, 'cameron_diaz': 23, 
		   'elvis_presley': 38, 'elton_john': 37, 'eric_clapton': 40, 
		   'thomas_edison': 94, 'john_cena': 55, 'gerald_ford': 44, 
		   'marvin_gaye': 70, 'taylor_swift': 92, 'reese_witherspoon': 84, 
		   'amy_schumer': 5, 'matt_damon': 71, 'brad_pitt': 19, 
		   'audrey_hepburn': 11, 'emily_blunt': 39, 'judy_garland': 59, 
		   'benedict_cumberbatch': 14, 'nicole_kidman': 78, 'david_beckham': 32, 
		   'leonardo_dicaprio': 64, 'channing_tatum': 27, 'woodrow_wilson': 97, 
		   'george_bush': 43, 'louis_armstrong': 65, 'jessica_alba': 53, 
		   'saoirse_ronan': 87, 'joan_crawford': 54, 'katy_perry': 61, 
		   'charlie_chaplin': 28, 'jennifer_lawrence': 52, 'paul_mccartney': 83, 
		   'bruce_willis': 22, 'drew_barrymore': 35, 'scarlett_johansson': 88, 
		   'mariah_carey': 67, 'ariana_grande': 9}

classes = {str(v): k for k, v in classes.iteritems()}

model = load_model("kardashian3.h5")

file = open("preds.csv", "w")
file.write("image_label,celebrity_name\n")

for filename in os.listdir(val_data_dir):
	img = cv2.imread(os.path.join(val_data_dir, filename))
	img = cv2.resize(img, (160, 160))
	img = np.reshape(img, [1, 160, 160, 3])
	pred = model.predict(img)
	pred = np.argmax(pred)
	celeb = classes[str(pred)]
	print(celeb)
	file.write(str(filename) + "," + celeb + "\n")

file.close()