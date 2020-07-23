tensorflow: 1.13.1
keras:2.2.4
python: 3.6.8
numpy: 1.16.2
PIL:5.4.1
matplotlib: 3.0.3
scipy:1.1.0

Enter path of galaxy_zoo(images in numpy) in img.py line 9

1. run separately

	--- convert numpy into pdf  
	python img.py
	--- (wait until finish) pre training for determining single or multiple 
	python disc_pre_train.py
	--- (wait until finish) load pre trained weights and train
	python train.py
	--- (wait until finish) save predicted img to result.output folder (using weights from the most recent train process)
	--- copy and paste merged images into result/input
	python predict.py

2. run in once
	
	python main.py

	--- wait until finish
	--- copy and paste merged images into result/input
	python predict.py
